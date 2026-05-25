#include "GeneticAlgorithm.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Minimal LCG for deterministic GA without <random> overhead
std::uint32_t GeneticAlgorithm::lcg(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
float GeneticAlgorithm::lcgFloat(std::uint32_t& s) {
    return (float)(lcg(s) >> 8) / (float)(1u << 24);
}

void GeneticAlgorithm::seedFrom(const std::vector<float>& champion, size_t n, unsigned seed, float sigma) {
    seed_        = seed;
    weightCount_ = champion.size();
    generation_  = 0;
    std::uint32_t rng = seed;
    population_.resize(n);
    population_[0].weights = champion;
    population_[0].fitness = 0.f;
    for (size_t i = 1; i < n; ++i) {
        population_[i].weights = champion;
        population_[i].fitness = 0.f;
        for (auto& w : population_[i].weights) {
            float u1 = lcgFloat(rng) + 1e-9f;
            float u2 = lcgFloat(rng);
            float noise = std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * 3.14159265f * u2);
            w += sigma * noise;
        }
    }
}

void GeneticAlgorithm::initPopulation(size_t n, size_t weightCount, unsigned seed) {
    seed_        = seed;
    weightCount_ = weightCount;
    generation_  = 0;
    std::uint32_t rng = seed;
    population_.resize(n);
    for (auto& g : population_) {
        g.weights.resize(weightCount);
        for (auto& w : g.weights)
            w = lcgFloat(rng) * 2.f - 1.f; // uniform [-1, 1]
        g.fitness = 0.f;
    }
}

size_t GeneticAlgorithm::tournamentSelect(std::uint32_t& rng, size_t k) {
    if (population_.empty()) throw std::runtime_error("GA: empty population");
    size_t best = lcg(rng) % population_.size();
    for (size_t i = 1; i < k; ++i) {
        size_t idx = lcg(rng) % population_.size();
        if (population_[idx].fitness > population_[best].fitness) best = idx;
    }
    return best;
}

Genome GeneticAlgorithm::crossover(const Genome& a, const Genome& b, std::uint32_t& rng) {
    Genome child;
    size_t n = a.weights.size();
    child.weights.resize(n);
    // Uniform crossover: each weight independently inherited from A or B
    for (size_t i = 0; i < n; ++i)
        child.weights[i] = (lcg(rng) & 1u) ? a.weights[i] : b.weights[i];
    return child;
}

void GeneticAlgorithm::mutate(Genome& g, float rate, float sigma, std::uint32_t& rng) {
    for (auto& w : g.weights) {
        if (lcgFloat(rng) < rate) {
            // Box-Muller approximation for Gaussian noise
            float u1 = lcgFloat(rng) + 1e-9f;
            float u2 = lcgFloat(rng);
            float n  = std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * 3.14159265f * u2);
            w += sigma * n;
        }
    }
}

void GeneticAlgorithm::evolve() {
    if (population_.empty()) return;
    size_t n = population_.size();
    std::uint32_t rng = seed_ ^ (std::uint32_t)(generation_ * 2654435761u);

    // Elitism: keep best 10%
    size_t eliteCount = std::max<size_t>(1, n / 10);
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    std::partial_sort(indices.begin(), indices.begin() + eliteCount, indices.end(),
        [&](size_t a, size_t b){ return population_[a].fitness > population_[b].fitness; });

    std::vector<Genome> next;
    next.reserve(n);
    for (size_t i = 0; i < eliteCount; ++i)
        next.push_back(population_[indices[i]]);

    // Random immigrants: 5% of population replaced with fresh random genomes
    size_t immigrantCount = std::max<size_t>(1, n / 20);

    // Fill rest via crossover + mutation
    while (next.size() < n) {
        Genome child;
        if (next.size() >= n - immigrantCount) {
            // Immigrant: completely random weights in [-1, 1]
            child.weights.resize(weightCount_);
            for (auto& w : child.weights)
                w = lcgFloat(rng) * 2.f - 1.f;
        } else {
            size_t pA = tournamentSelect(rng);
            size_t pB = tournamentSelect(rng);
            child = crossover(population_[pA], population_[pB], rng);
            mutate(child, 0.2f, 0.5f, rng); // 20% rate, sigma=0.5
        }
        child.fitness = 0.f;
        next.push_back(std::move(child));
    }

    population_ = std::move(next);
    ++generation_;
}
