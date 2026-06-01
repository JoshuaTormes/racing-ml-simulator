#include "training/Trainers.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ---- GeneticTrainer ----

void GeneticTrainer::init(size_t popSize, size_t weightCount, unsigned seed) {
    ga_.initPopulation(popSize, weightCount, seed);
}

void GeneticTrainer::initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) {
    ga_.seedFrom(champion, popSize, seed);
}

size_t GeneticTrainer::populationSize() const {
    return ga_.genomes().size();
}

const std::vector<float>& GeneticTrainer::weights(size_t i) const {
    return ga_.genomes()[i].weights;
}

void GeneticTrainer::setFitness(size_t i, float f) {
    ga_.setFitness(i, f);
}

void GeneticTrainer::evolve() {
    ga_.evolve();
}

int GeneticTrainer::generation() const {
    return ga_.generation();
}

// ---- RandomSearchTrainer helpers ----

std::uint32_t RandomSearchTrainer::lcg(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

float RandomSearchTrainer::lcgFloat(std::uint32_t& s) {
    return (float)(lcg(s) >> 8) / (float)(1u << 24);
}

void RandomSearchTrainer::fillRandom(std::vector<float>& w, std::uint32_t& rng) {
    for (auto& x : w) x = lcgFloat(rng) * 2.f - 1.f;
}

// ---- RandomSearchTrainer ----

void RandomSearchTrainer::init(size_t popSize, size_t weightCount, unsigned seed) {
    seed_ = seed;
    wc_   = weightCount;
    gen_  = 0;
    bestF_ = -std::numeric_limits<float>::infinity();
    bestW_.assign(weightCount, 0.f);

    std::uint32_t rng = seed;
    pop_.resize(popSize);
    fitness_.assign(popSize, 0.f);
    for (auto& w : pop_) {
        w.resize(weightCount);
        fillRandom(w, rng);
    }
}

void RandomSearchTrainer::initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) {
    seed_ = seed;
    wc_   = champion.size();
    gen_  = 0;
    bestF_ = -std::numeric_limits<float>::infinity();
    bestW_ = champion;

    std::uint32_t rng = seed;
    pop_.resize(popSize);
    fitness_.assign(popSize, 0.f);
    pop_[0] = champion;
    for (size_t i = 1; i < popSize; ++i) {
        pop_[i].resize(wc_);
        fillRandom(pop_[i], rng);
    }
}

void RandomSearchTrainer::evolve() {
    // Update global best
    for (size_t i = 0; i < pop_.size(); ++i) {
        if (fitness_[i] > bestF_) {
            bestF_ = fitness_[i];
            bestW_ = pop_[i];
        }
    }
    ++gen_;
    std::uint32_t rng = seed_ ^ (std::uint32_t)(gen_ * 2654435761u);
    // Elite in slot 0, rest random
    pop_[0] = bestW_;
    fitness_[0] = 0.f;
    for (size_t i = 1; i < pop_.size(); ++i) {
        fillRandom(pop_[i], rng);
        fitness_[i] = 0.f;
    }
}

// ---- HillClimbTrainer helpers ----

std::uint32_t HillClimbTrainer::lcg(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

float HillClimbTrainer::lcgFloat(std::uint32_t& s) {
    return (float)(lcg(s) >> 8) / (float)(1u << 24);
}

float HillClimbTrainer::gaussNoise(std::uint32_t& rng) {
    float u1 = lcgFloat(rng) + 1e-9f;
    float u2 = lcgFloat(rng);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * 3.14159265f * u2);
}

void HillClimbTrainer::rebuild() {
    std::uint32_t rng = seed_ ^ (std::uint32_t)(gen_ * 2654435761u);
    pop_[0] = cur_;
    for (size_t i = 1; i < pop_.size(); ++i) {
        pop_[i] = cur_;
        for (auto& w : pop_[i])
            w += sigma_ * gaussNoise(rng);
    }
    std::fill(fitness_.begin(), fitness_.end(), 0.f);
}

// ---- HillClimbTrainer ----

void HillClimbTrainer::init(size_t popSize, size_t weightCount, unsigned seed) {
    seed_  = seed;
    wc_    = weightCount;
    gen_   = 0;
    curF_  = -std::numeric_limits<float>::infinity();

    std::uint32_t rng = seed;
    cur_.resize(weightCount);
    for (auto& w : cur_) w = lcgFloat(rng) * 2.f - 1.f;

    pop_.resize(popSize);
    fitness_.assign(popSize, 0.f);
    for (auto& w : pop_) w.resize(weightCount);
    rebuild();
}

void HillClimbTrainer::initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) {
    seed_ = seed;
    wc_   = champion.size();
    gen_  = 0;
    curF_ = -std::numeric_limits<float>::infinity();
    cur_  = champion;

    pop_.resize(popSize);
    fitness_.assign(popSize, 0.f);
    for (auto& w : pop_) w.resize(wc_);
    rebuild();
}

void HillClimbTrainer::evolve() {
    // Find best of generation
    size_t bestIdx = 0;
    float  bestF   = fitness_[0];
    for (size_t i = 1; i < fitness_.size(); ++i) {
        if (fitness_[i] > bestF) { bestF = fitness_[i]; bestIdx = i; }
    }
    if (bestF > curF_) {
        curF_ = bestF;
        cur_  = pop_[bestIdx];
    }
    ++gen_;
    rebuild();
}

// ---- Factory ----

std::unique_ptr<Trainer> makeTrainer(const std::string& algo) {
    if (algo == "genetic")       return std::make_unique<GeneticTrainer>();
    if (algo == "random_search") return std::make_unique<RandomSearchTrainer>();
    if (algo == "hillclimb")     return std::make_unique<HillClimbTrainer>();
    throw std::runtime_error("Unknown trainer algo: " + algo);
}
