#include "Training.h"
#include "TrainingMath.h"
#include "NeuralNetwork.h"
#include "Car.h"
#include "core/Constants.h"
#include "core/TrackGen.h"
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <thread>
#include <atomic>
#include <numeric>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
TrainingSession::TrainingSession(SimConfig sim,
                                 std::unique_ptr<Trainer> trainer,
                                 int generations,
                                 std::string outDir,
                                 std::vector<std::string> trainMaps,
                                 std::vector<std::string> valMaps,
                                 MultiMapConfig mmCfg,
                                 const std::vector<float>* resumeChampion,
                                 bool logCsv,
                                 std::vector<std::string> testMaps)
    : game_(sim)
    , trainer_(std::move(trainer))
    , generations_(generations)
    , outDir_(std::move(outDir))
    , bestGlobalF_(-std::numeric_limits<float>::infinity())
    , trainMaps_(std::move(trainMaps))
    , valMaps_(std::move(valMaps))
    , testMaps_(std::move(testMaps))
    , mmCfg_(mmCfg)
    , bestValScore_(-std::numeric_limits<float>::infinity())
    , logCsv_(logCsv)
{
    if (trainMaps_.empty())
        trainMaps_.push_back(sim.map);

    NeuralNetwork probe(defaultTopology());
    size_t weightCount = probe.getWeights().size();

    if (resumeChampion)
        trainer_->initFromWeights(*resumeChampion, (size_t)sim.population, sim.seed);
    else
        trainer_->init((size_t)sim.population, weightCount, sim.seed);

    // Base training tracks from file paths.
    trainTracks_.reserve(trainMaps_.size());
    for (const auto& path : trainMaps_)
        trainTracks_.push_back(std::make_unique<Track>(path));

    // Static augmentation: expand the training set with cheap transforms of each base
    // map (mirror / reverse / width-scale). Variants become additional training maps.
    if (!mmCfg_.augment.empty()) {
        std::vector<TrackData> baseData;
        baseData.reserve(trainMaps_.size());
        for (const auto& path : trainMaps_)
            baseData.push_back(Track::loadData(path));
        for (const auto& base : baseData) {
            for (const auto& tok : mmCfg_.augment) {
                TrackData d;
                if      (tok == "mirror")              d = trackgen::mirrorX(base);
                else if (tok == "reverse")             d = trackgen::reverse(base);
                else if (tok.rfind("width:", 0) == 0)  d = trackgen::scaleWidth(base, std::stof(tok.substr(6)));
                else throw std::runtime_error("Unknown --augment token: " + tok);
                if (!mmCfg_.dumpGenMaps.empty())
                    Track::saveData(d, mmCfg_.dumpGenMaps + "/" + d.name + ".json");
                trainMaps_.push_back(d.name);
                trainTracks_.push_back(std::make_unique<Track>(d));
            }
        }
    }

    // Procedural training tracks (deterministic from the training seed).
    for (int i = 0; i < mmCfg_.proceduralTrain; ++i) {
        unsigned s = sim.seed * 2654435761u + 0x00C0FFEEu + (unsigned)i;
        TrackData d = trackgen::generateLoop(s, mmCfg_.genParams);
        if (!mmCfg_.dumpGenMaps.empty())
            Track::saveData(d, mmCfg_.dumpGenMaps + "/" + d.name + ".json");
        trainMaps_.push_back(d.name);
        trainTracks_.push_back(std::make_unique<Track>(d));
    }

    // Size per-map buffers to the final (possibly augmented) training set.
    perMapFitness_.assign(trainMaps_.size(),
                          std::vector<float>(trainer_->populationSize(), 0.f));
    perMapDoneReasons_.assign(trainMaps_.size(),
                              std::vector<DoneReason>(trainer_->populationSize(), DoneReason::None));

    valTracks_.reserve(valMaps_.size());
    for (const auto& path : valMaps_)
        valTracks_.push_back(std::make_unique<Track>(path));

    // Procedural validation tracks (deterministic from the training seed).
    for (int i = 0; i < mmCfg_.proceduralVal; ++i) {
        unsigned s = sim.seed * 40503u + 0x00BADA55u + (unsigned)i;
        TrackData d = trackgen::generateLoop(s, mmCfg_.genParams);
        valMaps_.push_back(d.name);
        valTracks_.push_back(std::make_unique<Track>(d));
    }

    testTracks_.reserve(testMaps_.size());
    for (const auto& path : testMaps_)
        testTracks_.push_back(std::make_unique<Track>(path));

    workerCount_ = sim.threads > 0
        ? sim.threads
        : (int)std::thread::hardware_concurrency();
    if (workerCount_ < 1) workerCount_ = 1;

    diagReasons_.assign(trainer_->populationSize(), DoneReason::None);

    std::filesystem::create_directories(outDir_);
}

// ---------------------------------------------------------------------------
// setMap
// ---------------------------------------------------------------------------
void TrainingSession::setMap(const std::string& path) {
    if (trainMaps_.size() == 1)
        trainMaps_[0] = path;
    beginGeneration();
}

// ---------------------------------------------------------------------------
// Multi-map helpers
// ---------------------------------------------------------------------------
void TrainingSession::loadTrainMap(size_t idx) {
    game_.loadMap(*trainTracks_[idx]);
}

void TrainingSession::recordCurrentMapFitness() {
    perMapFitness_[(size_t)currentMapInGen_] = game_.fitnesses();
}

std::vector<float> TrainingSession::aggregateFitness(const std::vector<int>& activeIdx) const {
    size_t P = trainer_->populationSize();
    size_t M = activeIdx.size();
    std::vector<float> result(P, 0.f);
    if (M == 0) return result;

    const FitnessAgg  agg   = mmCfg_.fitnessAgg;
    const float       alpha = mmCfg_.cvarAlpha;
    const float       wProg = game_.config().reward.w_progress;

    // Helper: per-map weight (default 1.0 if mapWeights is unset or too short)
    auto mapWeight = [&](size_t i) -> float {
        return (i < mmCfg_.mapWeights.size()) ? mmCfg_.mapWeights[i] : 1.0f;
    };

    if (agg == FitnessAgg::CVaRRank) {
        // Compute normalized ranks per active map, apply per-map weight.
        std::vector<std::vector<float>> mapRanks(M);
        for (size_t i = 0; i < M; ++i) {
            size_t mi = (size_t)activeIdx[i];
            mapRanks[i] = training_math::normalized_ranks(perMapFitness_[mi]);
            float w = mapWeight(mi);
            if (w != 1.0f)
                for (auto& v : mapRanks[i]) v *= w;
        }
        for (size_t c = 0; c < P; ++c) {
            std::vector<float> carRanks(M);
            for (size_t i = 0; i < M; ++i)
                carRanks[i] = mapRanks[i][c];
            result[c] = training_math::cvar(carRanks, alpha);
        }
    } else {
        std::vector<std::vector<float>> norm(M);
        for (size_t i = 0; i < M; ++i) {
            size_t mi = (size_t)activeIdx[i];
            switch (mmCfg_.mapNorm) {
                case MapNormMode::ZScore:
                    norm[i] = training_math::normalize_zscore(perMapFitness_[mi]);
                    break;
                case MapNormMode::MinMax:
                    norm[i] = training_math::normalize_minmax(perMapFitness_[mi]);
                    break;
                case MapNormMode::Progress:
                    norm[i] = training_math::normalize_progress(perMapFitness_[mi], wProg);
                    break;
            }
        }
        for (size_t c = 0; c < P; ++c) {
            if (agg == FitnessAgg::Min) {
                float worst = std::numeric_limits<float>::max();
                for (size_t i = 0; i < M; ++i)
                    worst = std::min(worst, norm[i][c]);
                result[c] = worst;
            } else if (agg == FitnessAgg::Mean) {
                float sum = 0.f;
                for (size_t i = 0; i < M; ++i) sum += norm[i][c];
                result[c] = sum / (float)M;
            } else { // CVaRRaw
                std::vector<float> vals(M);
                for (size_t i = 0; i < M; ++i) vals[i] = norm[i][c];
                result[c] = training_math::cvar(vals, alpha);
            }
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

// Decorrelated per-episode seed (deterministic given the training seed + indices).
static unsigned mixSeed(unsigned a, unsigned b, unsigned c, unsigned d, unsigned e) {
    unsigned h = a + 0x9e3779b9u;
    for (unsigned v : {b, c, d, e})
        h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
}

Game::EpisodeResult TrainingSession::evalCar(const Track& track, const NeuralNetwork& net,
                                             const RewardConfig& reward,
                                             size_t mapIdx, size_t carIdx) const {
    int E = std::max(1, mmCfg_.episodesPerEval);
    NeuralNetworkController ctrl(net); // copies net; scratch buffers stay caller-local
    Game::EpisodeResult rep{};
    float aggF = (mmCfg_.episodeAgg == EpisodeAgg::Min)
                    ? std::numeric_limits<float>::infinity() : 0.f;
    for (int e = 0; e < E; ++e) {
        EpisodeConfig ep;
        ep.seed        = mixSeed(game_.config().seed, (unsigned)currentGen_,
                                 (unsigned)mapIdx, (unsigned)carIdx, (unsigned)e);
        ep.randomSpawn = mmCfg_.randomSpawn;
        ep.sensorNoise = mmCfg_.sensorNoise;
        auto r = Game::simulateEpisode(track, ctrl, reward, ep);
        if (e == 0) rep = r; // representative maxProgress/doneReason/episodeTime
        if (mmCfg_.episodeAgg == EpisodeAgg::Min) aggF = std::min(aggF, r.fitness);
        else                                       aggF += r.fitness;
    }
    rep.fitness = (mmCfg_.episodeAgg == EpisodeAgg::Min) ? aggF : aggF / (float)E;
    return rep;
}

// ---------------------------------------------------------------------------
// Core generation loop
// ---------------------------------------------------------------------------
void TrainingSession::beginGeneration() {
    std::vector<std::unique_ptr<AIController>> ctrls;
    ctrls.reserve(trainer_->populationSize());
    for (size_t i = 0; i < trainer_->populationSize(); ++i) {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(i));
        ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
    }
    game_.setControllers(std::move(ctrls));

    for (auto& v : perMapFitness_)
        std::fill(v.begin(), v.end(), 0.f);
    currentMapInGen_ = 0;
    loadTrainMap(0);
}

void TrainingSession::tick() {
    game_.tick();
    advanceIfMapDone();
}

bool TrainingSession::generationComplete() const {
    return currentMapInGen_ == (int)trainMaps_.size() - 1
        && game_.episodeDone();
}

// ---------------------------------------------------------------------------
// evaluateGenerationParallel
// ---------------------------------------------------------------------------
void TrainingSession::evaluateGenerationParallel() {
    size_t P = trainer_->populationSize();
    size_t M = trainTracks_.size();
    auto activeIdx = training_math::active_map_indices(currentGen_, (int)M, mmCfg_.curriculum);
    size_t AM = activeIdx.size();

    std::vector<NeuralNetwork> nets;
    nets.reserve(P);
    for (size_t c = 0; c < P; ++c) {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(c));
        nets.push_back(std::move(nn));
    }

    diagReasons_.assign(P, DoneReason::None);
    perMapDoneReasons_.assign(M, std::vector<DoneReason>(P, DoneReason::None));

    // Zero out inactive maps
    {
        std::vector<bool> isActive(M, false);
        for (int mi : activeIdx) isActive[(size_t)mi] = true;
        for (size_t mi = 0; mi < M; ++mi)
            if (!isActive[mi])
                std::fill(perMapFitness_[mi].begin(), perMapFitness_[mi].end(), 0.f);
    }

    const RewardConfig& reward = game_.config().reward;
    const float progFrac = mmCfg_.progressiveFrac;

    if (progFrac >= 1.0f || AM <= 1) {
        // Standard path: evaluate all agents on all active maps.
        std::atomic<size_t> next{0};
        const size_t total = AM * P;
        std::vector<std::thread> workers;
        workers.reserve((size_t)workerCount_);
        for (int w = 0; w < workerCount_; ++w) {
            workers.emplace_back([&]() {
                size_t t;
                while ((t = next.fetch_add(1, std::memory_order_relaxed)) < total) {
                    size_t ai = t / P;
                    size_t c  = t % P;
                    size_t mi = (size_t)activeIdx[ai];
                    auto r = evalCar(*trainTracks_[mi], nets[c], reward, mi, c);
                    perMapFitness_[mi][c]     = r.fitness;
                    perMapDoneReasons_[mi][c] = r.doneReason;
                    if (mi == 0) diagReasons_[c] = r.doneReason;
                }
            });
        }
        for (auto& th : workers) th.join();
    } else {
        // Progressive path: pass 1 = map[activeIdx[0]] for full pop; then top-K on rest.
        size_t firstMap = (size_t)activeIdx[0];

        // Pass 1
        {
            std::atomic<size_t> next{0};
            std::vector<std::thread> workers;
            workers.reserve((size_t)workerCount_);
            for (int w = 0; w < workerCount_; ++w) {
                workers.emplace_back([&]() {
                    size_t c;
                    while ((c = next.fetch_add(1, std::memory_order_relaxed)) < P) {
                        auto r = evalCar(*trainTracks_[firstMap], nets[c], reward, firstMap, c);
                        perMapFitness_[firstMap][c]     = r.fitness;
                        perMapDoneReasons_[firstMap][c] = r.doneReason;
                        diagReasons_[c] = r.doneReason;
                    }
                });
            }
            for (auto& th : workers) th.join();
        }

        // Rank by pass-1 fitness; select top-K
        size_t K = (size_t)std::ceil(progFrac * (float)P);
        if (K < 1) K = 1;
        if (K > P) K = P;

        std::vector<size_t> order(P);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + (long)K, order.end(),
                          [&](size_t a, size_t b) {
                              return perMapFitness_[firstMap][a] > perMapFitness_[firstMap][b];
                          });
        std::vector<size_t> topK(order.begin(), order.begin() + (long)K);

        // Zero remaining maps for non-top agents
        for (size_t ai = 1; ai < AM; ++ai) {
            size_t mi = (size_t)activeIdx[ai];
            std::fill(perMapFitness_[mi].begin(), perMapFitness_[mi].end(), 0.f);
        }

        // Pass 2: top-K × remaining maps
        if (AM > 1) {
            const size_t tasks = (AM - 1) * K;
            std::atomic<size_t> next{0};
            std::vector<std::thread> workers;
            workers.reserve((size_t)workerCount_);
            for (int w = 0; w < workerCount_; ++w) {
                workers.emplace_back([&]() {
                    size_t t;
                    while ((t = next.fetch_add(1, std::memory_order_relaxed)) < tasks) {
                        size_t ai = 1 + t / K;
                        size_t ki = t % K;
                        size_t mi = (size_t)activeIdx[ai];
                        size_t c  = topK[ki];
                        auto r = evalCar(*trainTracks_[mi], nets[c], reward, mi, c);
                        perMapFitness_[mi][c]     = r.fitness;
                        perMapDoneReasons_[mi][c] = r.doneReason;
                    }
                });
            }
            for (auto& th : workers) th.join();
        }
    }
}

// ---------------------------------------------------------------------------
// finalizeGeneration
// ---------------------------------------------------------------------------
void TrainingSession::finalizeGeneration() {
    auto activeIdx = training_math::active_map_indices(currentGen_, (int)trainMaps_.size(), mmCfg_.curriculum);
    int activeMaps = (int)activeIdx.size();
    auto agg = aggregateFitness(activeIdx);
    size_t n = agg.size();
    for (size_t i = 0; i < n; ++i)
        trainer_->setFitness(i, agg[i]);

    // ---- Stats ---------------------------------------------------------------
    GenerationStats stats;
    stats.generation    = currentGen_ + 1;
    stats.population    = (int)n;
    stats.activeMapCount = activeMaps;

    // Sorted copy for percentiles
    std::vector<float> sortedAgg = agg;
    std::sort(sortedAgg.begin(), sortedAgg.end());

    float best = -std::numeric_limits<float>::infinity();
    float sum  = 0.f;
    size_t bestIdx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (agg[i] > best) { best = agg[i]; bestIdx = i; }
        sum += agg[i];
    }
    stats.aggBest = best;
    stats.aggMean = (n > 0) ? sum / (float)n : 0.f;
    stats.aggMin  = (n > 0) ? sortedAgg[0] : 0.f;

    // Median
    if (n > 0) {
        if (n % 2 == 1)
            stats.aggMedian = sortedAgg[n / 2];
        else
            stats.aggMedian = (sortedAgg[n/2 - 1] + sortedAgg[n/2]) * 0.5f;
    }

    // Std
    float var = 0.f;
    for (size_t i = 0; i < n; ++i) { float d = agg[i] - stats.aggMean; var += d*d; }
    stats.aggStd = (n > 0) ? std::sqrt(var / (float)n) : 0.f;

    // Top decile (mean of top 10%)
    size_t topK = std::max((size_t)1, (size_t)std::ceil(0.1f * (float)n));
    topK = std::min(topK, n);
    float topSum = 0.f;
    for (size_t i = n - topK; i < n; ++i) topSum += sortedAgg[i];
    stats.aggTopDecile = (topK > 0) ? topSum / (float)topK : 0.f;

    // Per-map stats
    float wProg = game_.config().reward.w_progress;
    std::vector<bool> isActiveMap(trainMaps_.size(), false);
    for (int i : activeIdx) isActiveMap[(size_t)i] = true;
    stats.perMap.resize(trainMaps_.size());
    for (size_t mi = 0; mi < trainMaps_.size(); ++mi) {
        auto& pm = stats.perMap[mi];
        pm.mapIndex  = (int)mi;
        pm.active    = isActiveMap[mi];

        if (pm.active) {
            pm.nEvaluated = (int)n;
            const auto& raw = perMapFitness_[mi];

            float rsum = 0.f, rbest = -std::numeric_limits<float>::infinity();
            for (float v : raw) { rsum += v; if (v > rbest) rbest = v; }
            pm.bestRaw = rbest;
            pm.meanRaw = rsum / (float)n;
            pm.normalizedBest = rbest / std::max(1.f, wProg);

            std::vector<float> sraw = raw;
            std::sort(sraw.begin(), sraw.end());
            pm.medianRaw = (n % 2 == 1) ? sraw[n/2] : (sraw[n/2-1] + sraw[n/2]) * 0.5f;

            float rv = 0.f;
            for (float v : raw) { float d = v - pm.meanRaw; rv += d*d; }
            pm.stdRaw = (n > 0) ? std::sqrt(rv / (float)n) : 0.f;

            for (size_t c = 0; c < n; ++c) {
                switch (perMapDoneReasons_[mi][c]) {
                    case DoneReason::Completed: ++pm.nCompleted; break;
                    case DoneReason::Collision: ++pm.nCollision; break;
                    case DoneReason::Stall:     ++pm.nStall;     break;
                    case DoneReason::Timeout:   ++pm.nTimeout;   break;
                    default: break;
                }
            }
        }
    }

    // ---- Save snapshot -------------------------------------------------------
    {
        std::ostringstream fname;
        fname << outDir_ << "/gen_" << std::setw(4) << std::setfill('0') << stats.generation << ".rnnw";
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(bestIdx));
        nn.save(fname.str());
    }
    // best.rnnw selection. Default (legacy): by raw train fitness (CVaRRank uses
    // the primary-map raw score, since rank-aggregated scores saturate at 1.0).
    // When selectByVal is enabled and validation maps exist, the champion is chosen
    // by validation performance instead — handled in the held-out block below.
    const bool valSelection = mmCfg_.selectByVal && !valTracks_.empty();
    if (!valSelection) {
        float saveCriterion = (mmCfg_.fitnessAgg == FitnessAgg::CVaRRank && !perMapFitness_.empty())
            ? perMapFitness_[0][bestIdx]
            : best;
        if (saveCriterion > bestGlobalF_) {
            bestGlobalF_ = saveCriterion;
            NeuralNetwork nn(defaultTopology());
            nn.setWeights(trainer_->weights(bestIdx));
            nn.save(outDir_ + "/best.rnnw");
        }
    }

    printStats(stats);
    if (logCsv_) writeTrainingCsvRow(stats);

    lastStats_ = stats;
    history_.push_back(stats);

    // ---- Held-out validation / test ------------------------------------------
    bool isLastGen   = (currentGen_ + 1 == generations_);
    bool wantHeldOut = !valMaps_.empty() || !testMaps_.empty();
    bool isValGen    = wantHeldOut && ((currentGen_ + 1) % valEvery_ == 0 || isLastGen);
    if (isValGen) {
        // Champion to report (and, under selectByVal, to save as best.rnnw):
        // chosen by validation when enabled, else the train-best.
        size_t champ = valSelection ? selectChampionByVal(agg) : bestIdx;
        evaluateHeldOut(champ);
    }

    trainer_->evolve();
    ++currentGen_;
}

void TrainingSession::endGeneration() {
    recordCurrentMapFitness();

    // Capture done-reasons for the windowed path (sequential map visits)
    // Use diagReasons_ from the first map to stay consistent with parallel path.
    // For all maps, pull from game_.cars() for the current (last) map.
    size_t nCars = game_.cars().size();
    size_t M = trainMaps_.size();
    if (perMapDoneReasons_.size() < M)
        perMapDoneReasons_.assign(M, std::vector<DoneReason>(nCars, DoneReason::None));
    for (size_t c = 0; c < nCars; ++c)
        perMapDoneReasons_[(size_t)currentMapInGen_][c] = game_.cars()[c].doneReason;

    diagReasons_.resize(nCars);
    for (size_t c = 0; c < nCars; ++c)
        diagReasons_[c] = perMapDoneReasons_[0][c];

    finalizeGeneration();
}

bool TrainingSession::done() const {
    return currentGen_ >= generations_;
}

// ---------------------------------------------------------------------------
// Held-out validation / test
// ---------------------------------------------------------------------------
static std::string doneReasonStr(DoneReason r) {
    switch (r) {
        case DoneReason::Completed: return "completed";
        case DoneReason::Collision: return "collision";
        case DoneReason::Stall:     return "stall";
        case DoneReason::Timeout:   return "timeout";
        default:                    return "?";
    }
}

void TrainingSession::logChampionOnMaps(const char* label,
                                        const std::vector<std::unique_ptr<Track>>& tracks,
                                        const std::vector<std::string>& names,
                                        size_t championIdx,
                                        std::ofstream& log) {
    const auto& champWeights = trainer_->weights(championIdx);
    const RewardConfig& reward = game_.config().reward;

    if (logCsv_ && log.is_open())
        log << currentGen_ + 1;

    for (size_t vi = 0; vi < tracks.size(); ++vi) {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(champWeights);
        NeuralNetworkController ctrl(std::move(nn));
        auto r = Game::simulateEpisode(*tracks[vi], ctrl, reward);

        std::string reason = doneReasonStr(r.doneReason);
        auto mapName = std::filesystem::path(names[vi]).stem().string();
        std::cout << std::fixed << std::setprecision(3)
                  << "    " << label << "[" << mapName << "]: prog=" << r.maxProgress
                  << " reason=" << reason << "\n";

        if (logCsv_ && log.is_open())
            log << "," << r.maxProgress << "," << reason << "," << r.fitness;
    }

    if (logCsv_ && log.is_open()) {
        log << "\n";
        log.flush();
    }
}

void TrainingSession::evaluateHeldOut(size_t championIdx) {
    if (valMaps_.empty() && testMaps_.empty()) return;
    std::cout << "  [held-out gen=" << currentGen_ + 1 << "]\n";
    if (!valMaps_.empty())
        logChampionOnMaps("val", valTracks_, valMaps_, championIdx, heldOutLog_);
    if (!testMaps_.empty())
        logChampionOnMaps("test", testTracks_, testMaps_, championIdx, testLog_);
}

float TrainingSession::meanProgressOn(const std::vector<std::unique_ptr<Track>>& tracks,
                                      const std::vector<float>& weights) const {
    if (tracks.empty()) return 0.f;
    const RewardConfig& reward = game_.config().reward;
    float sum = 0.f;
    for (const auto& tr : tracks) {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(weights);
        NeuralNetworkController ctrl(std::move(nn));
        sum += Game::simulateEpisode(*tr, ctrl, reward).maxProgress;
    }
    return sum / (float)tracks.size();
}

size_t TrainingSession::selectChampionByVal(const std::vector<float>& agg) {
    size_t P = agg.size();
    if (P == 0) return 0;
    int K = std::max(1, mmCfg_.valSelectTopK);
    if ((size_t)K > P) K = (int)P;

    // Top-K genomes by aggregated train fitness (always includes the train-best).
    std::vector<size_t> order(P);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + K, order.end(),
                      [&](size_t a, size_t b) { return agg[a] > agg[b]; });

    // Among those candidates, keep the one with the best mean validation progress.
    size_t bestCand  = order[0];
    float  bestScore = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < K; ++i) {
        size_t c = order[(size_t)i];
        float s  = meanProgressOn(valTracks_, trainer_->weights(c));
        if (s > bestScore) { bestScore = s; bestCand = c; }
    }

    // Persist as best.rnnw only when it improves the validation high-water mark.
    if (bestScore > bestValScore_) {
        bestValScore_ = bestScore;
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(trainer_->weights(bestCand));
        nn.save(outDir_ + "/best.rnnw");
        std::cout << std::fixed << std::setprecision(3)
                  << "  [select-by-val] saved best.rnnw (val prog=" << bestScore << ")\n";
    }
    return bestCand;
}

// ---------------------------------------------------------------------------
// runAll (headless)
// ---------------------------------------------------------------------------
void TrainingSession::runAll() {
    if (logCsv_) {
        trainingLog_.open(outDir_ + "/training_log.csv", std::ios::trunc);
        writeTrainingCsvHeader();
        heldOutLog_.open(outDir_ + "/held_out_log.csv", std::ios::trunc);
        heldOutLog_ << "gen";
        for (size_t vi = 0; vi < valMaps_.size(); ++vi)
            heldOutLog_ << ",m" << vi << "_progress,m" << vi << "_done_reason,m" << vi << "_fitness_raw";
        heldOutLog_ << "\n";
        heldOutLog_.flush();

        if (!testMaps_.empty()) {
            testLog_.open(outDir_ + "/test_log.csv", std::ios::trunc);
            testLog_ << "gen";
            for (size_t ti = 0; ti < testMaps_.size(); ++ti)
                testLog_ << ",m" << ti << "_progress,m" << ti << "_done_reason,m" << ti << "_fitness_raw";
            testLog_ << "\n";
            testLog_.flush();
        }
    }

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
    // Agg mode string
    std::string aggMode;
    switch (mmCfg_.fitnessAgg) {
        case FitnessAgg::CVaRRank: {
            std::ostringstream ss;
            ss << "cvar-rank α=" << std::fixed << std::setprecision(2) << mmCfg_.cvarAlpha;
            aggMode = ss.str();
            break;
        }
        case FitnessAgg::Min:    aggMode = "min";    break;
        case FitnessAgg::Mean:   aggMode = "mean";   break;
        case FitnessAgg::CVaRRaw: {
            std::ostringstream ss;
            ss << "cvar-raw α=" << std::fixed << std::setprecision(2) << mmCfg_.cvarAlpha;
            aggMode = ss.str();
            break;
        }
    }

    std::cout << std::fixed << std::setprecision(3)
              << "gen " << std::setw(4) << std::setfill('0') << s.generation
              << std::setfill(' ') << "/" << generations_
              << "  agg[" << aggMode << "]:"
              << " best=" << std::setw(7) << s.aggBest
              << " mean=" << std::setw(7) << s.aggMean
              << " std="  << std::setw(6) << s.aggStd;

    // Under CVaRRank, show the raw fitness of the champion on map 0 as a progress proxy,
    // since rank-based agg scores are always in [0,1] and look flat on the console.
    if (mmCfg_.fitnessAgg == FitnessAgg::CVaRRank && !s.perMap.empty() && s.perMap[0].active)
        std::cout << std::fixed << std::setprecision(1)
                  << "  ref_best=" << std::setw(7) << s.perMap[0].bestRaw << " (m0 raw)";

    std::cout << "\n";

    std::cout << "  active maps: " << s.activeMapCount << "/" << (int)trainMaps_.size() << "\n";

    for (size_t mi = 0; mi < s.perMap.size(); ++mi) {
        const auto& pm = s.perMap[mi];
        auto stem = std::filesystem::path(trainMaps_[mi]).stem().string();
        // Truncate map name to 16 chars
        if ((int)stem.size() > 16) stem = stem.substr(0, 16);

        std::cout << "  m" << mi << "[" << std::left << std::setw(16) << stem << std::right << "]: ";
        if (pm.active) {
            std::cout << std::fixed << std::setprecision(1)
                      << "best=" << std::setw(7) << pm.bestRaw
                      << " mean=" << std::setw(7) << pm.meanRaw
                      << " med=" << std::setw(7) << pm.medianRaw
                      << " std=" << std::setw(6) << pm.stdRaw
                      << " | comp=" << std::setw(3) << pm.nCompleted
                      << " col=" << std::setw(3) << pm.nCollision
                      << " sta=" << std::setw(3) << pm.nStall
                      << " to=" << std::setw(3) << pm.nTimeout
                      << "\n";
        } else {
            std::cout << "---inactive--- (curriculum)\n";
        }
    }
}

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------
void TrainingSession::writeTrainingCsvHeader() {
    trainingLog_ << "gen,active_maps,agg_best,agg_mean,agg_median,agg_std,agg_top10,agg_min";
    for (size_t mi = 0; mi < trainMaps_.size(); ++mi)
        trainingLog_ << ",m" << mi << "_active"
                     << ",m" << mi << "_best_raw"
                     << ",m" << mi << "_mean_raw"
                     << ",m" << mi << "_median_raw"
                     << ",m" << mi << "_std_raw"
                     << ",m" << mi << "_normalized_best"
                     << ",m" << mi << "_completed"
                     << ",m" << mi << "_collision"
                     << ",m" << mi << "_stall"
                     << ",m" << mi << "_timeout";
    trainingLog_ << "\n";
    trainingLog_.flush();
}

void TrainingSession::writeTrainingCsvRow(const GenerationStats& s) {
    if (!trainingLog_.is_open()) return;
    trainingLog_ << std::fixed << std::setprecision(6)
                 << s.generation
                 << "," << s.activeMapCount
                 << "," << s.aggBest
                 << "," << s.aggMean
                 << "," << s.aggMedian
                 << "," << s.aggStd
                 << "," << s.aggTopDecile
                 << "," << s.aggMin;
    for (const auto& pm : s.perMap) {
        if (pm.active) {
            trainingLog_ << ",true"
                         << "," << pm.bestRaw
                         << "," << pm.meanRaw
                         << "," << pm.medianRaw
                         << "," << pm.stdRaw
                         << "," << pm.normalizedBest
                         << "," << pm.nCompleted
                         << "," << pm.nCollision
                         << "," << pm.nStall
                         << "," << pm.nTimeout;
        } else {
            trainingLog_ << ",false,,,,,,,,,";
        }
    }
    trainingLog_ << "\n";
    trainingLog_.flush();
}
