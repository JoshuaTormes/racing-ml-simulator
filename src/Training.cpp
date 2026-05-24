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

TrainingSession::TrainingSession(SimConfig sim,
                                 std::unique_ptr<Trainer> trainer,
                                 int generations,
                                 std::string outDir,
                                 const std::vector<float>* resumeChampion)
    : game_(sim)
    , trainer_(std::move(trainer))
    , generations_(generations)
    , outDir_(std::move(outDir))
    , bestGlobalF_(-std::numeric_limits<float>::infinity())
{
    NeuralNetwork probe({OBS_SIZE, 8, 2});
    size_t weightCount = probe.getWeights().size();

    if (resumeChampion)
        trainer_->initFromWeights(*resumeChampion, (size_t)sim.population, sim.seed);
    else
        trainer_->init((size_t)sim.population, weightCount, sim.seed);

    std::filesystem::create_directories(outDir_);
}

void TrainingSession::beginGeneration() {
    std::vector<std::unique_ptr<AIController>> ctrls;
    ctrls.reserve(trainer_->populationSize());
    for (size_t i = 0; i < trainer_->populationSize(); ++i) {
        NeuralNetwork nn({OBS_SIZE, 8, 2});
        nn.setWeights(trainer_->weights(i));
        ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
    }
    game_.setControllers(std::move(ctrls));
    game_.reset();
}

void TrainingSession::tick() {
    game_.tick();
}

bool TrainingSession::generationComplete() const {
    return game_.episodeDone();
}

void TrainingSession::endGeneration() {
    auto fitnesses = game_.fitnesses();
    size_t n = fitnesses.size();
    for (size_t i = 0; i < n; ++i)
        trainer_->setFitness(i, fitnesses[i]);

    // Stats
    GenerationStats stats;
    stats.generation = currentGen_ + 1;
    stats.population = (int)n;

    float best = -std::numeric_limits<float>::infinity();
    float sum  = 0.f;
    size_t bestIdx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (fitnesses[i] > best) { best = fitnesses[i]; bestIdx = i; }
        sum += fitnesses[i];
    }
    stats.bestFitness = best;
    stats.meanFitness = (n > 0) ? sum / (float)n : 0.f;

    float var = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float d = fitnesses[i] - stats.meanFitness;
        var += d * d;
    }
    stats.stdFitness = (n > 0) ? std::sqrt(var / (float)n) : 0.f;

    // Count completed cars
    for (const auto& car : game_.cars())
        if (car.doneReason == DoneReason::Completed) ++stats.completed;

    // Save generation snapshot
    {
        std::ostringstream fname;
        fname << outDir_ << "/gen_" << std::setw(4) << std::setfill('0') << stats.generation << ".rnnw";
        NeuralNetwork nn({OBS_SIZE, 8, 2});
        nn.setWeights(trainer_->weights(bestIdx));
        nn.save(fname.str());
    }

    // Update and save global best
    if (best > bestGlobalF_) {
        bestGlobalF_ = best;
        NeuralNetwork nn({OBS_SIZE, 8, 2});
        nn.setWeights(trainer_->weights(bestIdx));
        nn.save(outDir_ + "/best.rnnw");
    }

    printStats(stats);
    lastStats_ = stats;
    history_.push_back(stats);

    trainer_->evolve();
    ++currentGen_;
}

bool TrainingSession::done() const {
    return currentGen_ >= generations_;
}

void TrainingSession::runAll() {
    while (!done()) {
        beginGeneration();
        while (!generationComplete()) tick();
        endGeneration();
    }
}

void TrainingSession::printStats(const GenerationStats& s) const {
    std::cout << std::fixed << std::setprecision(2)
              << "gen " << std::setw(4) << s.generation
              << "/" << generations_
              << "  best=" << std::setw(8) << s.bestFitness
              << "  mean=" << std::setw(8) << s.meanFitness
              << "  std="  << std::setw(6) << s.stdFitness
              << "  done=" << s.completed << "/" << s.population
              << "\n";
}
