#include "Training.h"
#include "NeuralNetwork.h"
#include "Car.h"
#include "core/Constants.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <thread>
#include <atomic>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
TrainingSession::TrainingSession(SimConfig sim,
                                 std::unique_ptr<Trainer> trainer,
                                 int generations,
                                 std::string outDir,
                                 std::vector<std::string> trainMaps,
                                 std::vector<std::string> valMaps,
                                 FitnessAgg agg,
                                 const std::vector<float>* resumeChampion)
    : game_(sim)
    , trainer_(std::move(trainer))
    , generations_(generations)
    , outDir_(std::move(outDir))
    , bestGlobalF_(-std::numeric_limits<float>::infinity())
    , trainMaps_(std::move(trainMaps))
    , valMaps_(std::move(valMaps))
    , agg_(agg)
{
    // Fallback: if no train maps supplied, use the map from SimConfig
    if (trainMaps_.empty())
        trainMaps_.push_back(sim.map);

    NeuralNetwork probe(defaultTopology());
    size_t weightCount = probe.getWeights().size();

    if (resumeChampion)
        trainer_->initFromWeights(*resumeChampion, (size_t)sim.population, sim.seed);
    else
        trainer_->init((size_t)sim.population, weightCount, sim.seed);

    // Pre-size per-map fitness storage (filled fresh each generation)
    perMapFitness_.assign(trainMaps_.size(),
                          std::vector<float>(trainer_->populationSize(), 0.f));
    mapNorm_.assign(trainMaps_.size(), 1.f);

    // Cache training tracks (loaded once; reused every generation)
    trainTracks_.reserve(trainMaps_.size());
    for (size_t i = 0; i < trainMaps_.size(); ++i) {
        trainTracks_.push_back(std::make_unique<Track>(trainMaps_[i]));
        mapNorm_[i] = mapNormConst(*trainTracks_[i]);
    }

    // Cache validation tracks
    valTracks_.reserve(valMaps_.size());
    for (const auto& vmap : valMaps_)
        valTracks_.push_back(std::make_unique<Track>(vmap));

    // Worker thread count
    workerCount_ = sim.threads > 0
        ? sim.threads
        : (int)std::thread::hardware_concurrency();
    if (workerCount_ < 1) workerCount_ = 1;

    // Diagnostic reasons vector (one per car)
    diagReasons_.assign(trainer_->populationSize(), DoneReason::None);

    std::filesystem::create_directories(outDir_);
}

// ---------------------------------------------------------------------------
// setMap — public API for the windowed UI
// ---------------------------------------------------------------------------
void TrainingSession::setMap(const std::string& path) {
    if (trainMaps_.size() == 1) {
        // Single-map mode: honour the user's selection
        trainMaps_[0] = path;
    }
    // Multi-map mode: restart generation from trainMaps_[0] (path ignored).
    beginGeneration();
}

// ---------------------------------------------------------------------------
// Multi-map helpers
// ---------------------------------------------------------------------------
float TrainingSession::mapNormConst(const Track& t) const {
    int numWp = (int)t.waypoints().size();
    return std::max(1.f, game_.config().reward.w_progress * (float)(numWp - 1));
}

void TrainingSession::loadTrainMap(size_t idx) {
    game_.loadMap(trainMaps_[idx]);
    mapNorm_[idx] = mapNormConst(game_.track());
}

void TrainingSession::recordCurrentMapFitness() {
    perMapFitness_[(size_t)currentMapInGen_] = game_.fitnesses();
}

std::vector<float> TrainingSession::aggregateFitness() const {
    size_t n = trainer_->populationSize();
    size_t m = trainMaps_.size();
    std::vector<float> result(n);

    for (size_t c = 0; c < n; ++c) {
        if (agg_ == FitnessAgg::Min) {
            float worst = std::numeric_limits<float>::max();
            for (size_t mi = 0; mi < m; ++mi) {
                float score = perMapFitness_[mi][c] / mapNorm_[mi];
                worst = std::min(worst, score);
            }
            result[c] = worst;
        } else { // Mean
            float sum = 0.f;
            for (size_t mi = 0; mi < m; ++mi)
                sum += perMapFitness_[mi][c] / mapNorm_[mi];
            result[c] = sum / (float)m;
        }
    }
    return result;
}

void TrainingSession::advanceIfMapDone() {
    if (game_.episodeDone() && currentMapInGen_ + 1 < (int)trainMaps_.size()) {
        recordCurrentMapFitness();
        ++currentMapInGen_;
        loadTrainMap((size_t)currentMapInGen_);
    }
}

// ---------------------------------------------------------------------------
// Core generation loop
// ---------------------------------------------------------------------------
void TrainingSession::beginGeneration() {
    // Build controllers from current generation's weights
    std::vector<std::unique_ptr<AIController>> ctrls;
    ctrls.reserve(trainer_->populationSize());
    for (size_t i = 0; i < trainer_->populationSize(); ++i) {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(i));
        ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
    }
    game_.setControllers(std::move(ctrls));

    // Reset per-map storage and start at map 0
    for (auto& v : perMapFitness_)
        std::fill(v.begin(), v.end(), 0.f);
    currentMapInGen_ = 0;
    loadTrainMap(0); // loadMap -> spawnCars -> ctrl->reset (no-op for NN)
}

void TrainingSession::tick() {
    game_.tick();
    advanceIfMapDone();
}

bool TrainingSession::generationComplete() const {
    // True only when the LAST training map's episode is done
    return currentMapInGen_ == (int)trainMaps_.size() - 1
        && game_.episodeDone();
}

// ---------------------------------------------------------------------------
// evaluateGenerationParallel — fills perMapFitness_[M][P] using a worker pool
// ---------------------------------------------------------------------------
void TrainingSession::evaluateGenerationParallel() {
    size_t P = trainer_->populationSize();
    size_t M = trainTracks_.size();

    // Build one NeuralNetwork per car (read-only during workers; each worker
    // copies it into a local NeuralNetworkController to avoid data races)
    std::vector<NeuralNetwork> nets;
    nets.reserve(P);
    for (size_t c = 0; c < P; ++c) {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(c));
        nets.push_back(std::move(nn));
    }

    // diagReasons_ captures done-reason from map 0 for the stats breakdown
    diagReasons_.assign(P, DoneReason::None);

    std::atomic<size_t> next{0};
    const size_t total = M * P;
    const RewardConfig& reward = game_.config().reward;

    std::vector<std::thread> workers;
    workers.reserve((size_t)workerCount_);

    for (int w = 0; w < workerCount_; ++w) {
        workers.emplace_back([&]() {
            size_t t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < total) {
                size_t m = t / P;
                size_t c = t % P;
                // Copy the NN into a local controller (each task owns its copy)
                NeuralNetworkController ctrl(nets[c]);
                auto r = Game::simulateEpisode(*trainTracks_[m], ctrl, reward);
                perMapFitness_[m][c] = r.fitness;
                if (m == 0)
                    diagReasons_[c] = r.doneReason;
            }
        });
    }
    for (auto& th : workers) th.join();
}

// ---------------------------------------------------------------------------
// finalizeGeneration — aggregation, stats, saves, held-out, evolve
// (used by both the parallel headless path and the windowed serial path)
// ---------------------------------------------------------------------------
void TrainingSession::finalizeGeneration() {
    // Aggregate across all training maps
    auto agg = aggregateFitness();
    size_t n = agg.size();
    for (size_t i = 0; i < n; ++i)
        trainer_->setFitness(i, agg[i]);

    // ---- Stats ----------------------------------------------------------------
    GenerationStats stats;
    stats.generation = currentGen_ + 1;
    stats.population = (int)n;

    float best = -std::numeric_limits<float>::infinity();
    float sum  = 0.f;
    size_t bestIdx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (agg[i] > best) { best = agg[i]; bestIdx = i; }
        sum += agg[i];
    }
    stats.bestFitness = best;
    stats.meanFitness = (n > 0) ? sum / (float)n : 0.f;
    stats.aggScore    = best;

    float var = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float d = agg[i] - stats.meanFitness;
        var += d * d;
    }
    stats.stdFitness = (n > 0) ? std::sqrt(var / (float)n) : 0.f;

    // Best normalised score per training map (diagnostic)
    stats.perMapBest.resize(trainMaps_.size());
    for (size_t mi = 0; mi < trainMaps_.size(); ++mi) {
        float mb = -std::numeric_limits<float>::infinity();
        for (size_t c = 0; c < n; ++c)
            mb = std::max(mb, perMapFitness_[mi][c] / mapNorm_[mi]);
        stats.perMapBest[mi] = mb;
    }

    // Done-reason breakdown from diagReasons_ (map 0, one entry per car)
    for (size_t c = 0; c < n; ++c) {
        switch (diagReasons_[c]) {
            case DoneReason::Completed: ++stats.completed;  break;
            case DoneReason::Collision: ++stats.nCollision; break;
            case DoneReason::Stall:     ++stats.nStall;     break;
            case DoneReason::Timeout:   ++stats.nTimeout;   break;
            default: break;
        }
    }

    // ---- Save generation snapshot (by aggregated score) ----------------------
    {
        std::ostringstream fname;
        fname << outDir_ << "/gen_" << std::setw(4) << std::setfill('0') << stats.generation << ".rnnw";
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(bestIdx));
        nn.save(fname.str());
    }

    // Update and save global best
    if (best > bestGlobalF_) {
        bestGlobalF_ = best;
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(bestIdx));
        nn.save(outDir_ + "/best.rnnw");
    }

    printStats(stats);
    lastStats_ = stats;
    history_.push_back(stats);

    // ---- Held-out evaluation (every valEvery_ gens and at the last gen) ------
    bool isLastGen = (currentGen_ + 1 == generations_);
    bool isValGen  = (!valMaps_.empty()) && ((currentGen_ + 1) % valEvery_ == 0 || isLastGen);
    if (isValGen)
        evaluateHeldOut(bestIdx);

    trainer_->evolve();
    ++currentGen_;
}

void TrainingSession::endGeneration() {
    // Record last (current) map fitness — earlier maps are recorded in advanceIfMapDone
    recordCurrentMapFitness();

    // Capture diagReasons_ from game_.cars() (map 0 = first map; windowed visits sequentially,
    // but representative sample from the last-processed map is acceptable here —
    // we use the current map's cars directly to match the old behaviour for the windowed path)
    diagReasons_.resize(game_.cars().size());
    for (size_t c = 0; c < game_.cars().size(); ++c)
        diagReasons_[c] = game_.cars()[c].doneReason;

    finalizeGeneration();
}

bool TrainingSession::done() const {
    return currentGen_ >= generations_;
}

// ---------------------------------------------------------------------------
// Held-out validation — uses simulateEpisode; no game_.loadMap calls
// ---------------------------------------------------------------------------
void TrainingSession::evaluateHeldOut(size_t championIdx) {
    if (valMaps_.empty()) return;

    const auto& champWeights = trainer_->weights(championIdx);
    const RewardConfig& reward = game_.config().reward;

    std::cout << "  [held-out gen=" << currentGen_ + 1 << "]\n";
    for (size_t vi = 0; vi < valTracks_.size(); ++vi) {
        const Track& vtrack = *valTracks_[vi];
        int numWp = (int)vtrack.waypoints().size();

        NeuralNetwork nn(defaultTopology());
        nn.setWeights(champWeights);
        NeuralNetworkController ctrl(std::move(nn));

        auto r = Game::simulateEpisode(vtrack, ctrl, reward);

        float progFrac = (numWp > 1) ? r.maxProgress / (float)(numWp - 1) : 0.f;

        std::string reason;
        switch (r.doneReason) {
            case DoneReason::Completed: reason = "completed"; break;
            case DoneReason::Collision: reason = "collision"; break;
            case DoneReason::Stall:     reason = "stall";     break;
            case DoneReason::Timeout:   reason = "timeout";   break;
            default:                    reason = "?";         break;
        }

        auto mapName = std::filesystem::path(valMaps_[vi]).stem().string();
        std::cout << std::fixed << std::setprecision(3)
                  << "    val[" << mapName << "]: prog=" << progFrac
                  << " reason=" << reason << "\n";
    }
}

// ---------------------------------------------------------------------------
// runAll (headless) — parallel evaluation; game_ not used for evaluation
// ---------------------------------------------------------------------------
void TrainingSession::runAll() {
    while (!done()) {
        for (auto& v : perMapFitness_) std::fill(v.begin(), v.end(), 0.f);
        evaluateGenerationParallel();
        finalizeGeneration();
    }
}

// ---------------------------------------------------------------------------
// printStats
// ---------------------------------------------------------------------------
void TrainingSession::printStats(const GenerationStats& s) const {
    std::cout << std::fixed << std::setprecision(3)
              << "gen " << std::setw(4) << s.generation
              << "/" << generations_
              << "  agg=" << std::setw(7) << s.aggScore;

    // Per-map breakdown
    std::cout << " |";
    for (size_t mi = 0; mi < s.perMapBest.size(); ++mi)
        std::cout << " m" << mi << "=" << std::setw(6) << s.perMapBest[mi];
    std::cout << " |";

    std::cout << "  mean=" << std::setw(7) << s.meanFitness
              << "  std="  << std::setw(6) << s.stdFitness
              << "  done=" << s.completed << "/" << s.population
              << "  [col=" << s.nCollision
              << " stall=" << s.nStall
              << " timeout=" << s.nTimeout << "]"
              << "\n";
}
