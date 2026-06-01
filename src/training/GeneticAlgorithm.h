#pragma once
#include <vector>
#include <cstdint>

// Genome: flat weight vector + fitness score from the simulation
struct Genome {
    std::vector<float> weights;
    float              fitness = 0.f;
};

// Neuroevolution driver. Interface is complete; bodies are functional stubs
// ready for a full algorithm (NEAT, CMA-ES, etc.) to replace the internals.
class GeneticAlgorithm {
public:
    // Create a population of n genomes matching the given NN weight count
    void initPopulation(size_t n, size_t weightCount, unsigned seed = 0);

    const std::vector<Genome>& genomes() const { return population_; }

    // Called by the simulation after each episode
    void setFitness(size_t i, float f) { population_[i].fitness = f; }

    // Selection → crossover → mutation → new generation
    void evolve();

    int generation() const { return generation_; }

    // Seed population from a champion (copies + Gaussian noise); used for --load+--train
    void seedFrom(const std::vector<float>& champion, size_t n, unsigned seed, float sigma = 0.3f);

private:
    std::vector<Genome> population_;
    int                 generation_ = 0;
    unsigned            seed_       = 0;
    size_t              weightCount_= 0;

    // Tournament selection: returns index of winner from k random candidates
    size_t tournamentSelect(std::uint32_t& rng, size_t k = 2);

    // Uniform crossover: each weight independently taken from parent A or B
    static Genome crossover(const Genome& a, const Genome& b, std::uint32_t& rng);

    // Gaussian mutation: each weight mutated with probability rate
    static void mutate(Genome& g, float rate, float sigma, std::uint32_t& rng);

    static std::uint32_t lcg(std::uint32_t& state);
    static float         lcgFloat(std::uint32_t& state);
};
