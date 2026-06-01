#pragma once
#include "training/Trainer.h"
#include "training/GeneticAlgorithm.h"
#include <memory>
#include <string>
#include <cstdint>
#include <limits>

class GeneticTrainer : public Trainer {
public:
    void init(size_t popSize, size_t weightCount, unsigned seed) override;
    void initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) override;
    size_t populationSize() const override;
    const std::vector<float>& weights(size_t i) const override;
    void setFitness(size_t i, float f) override;
    void evolve() override;
    int  generation() const override;
    const char* name() const override { return "genetic"; }
private:
    GeneticAlgorithm ga_;
};

class RandomSearchTrainer : public Trainer {
public:
    void init(size_t popSize, size_t weightCount, unsigned seed) override;
    void initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) override;
    size_t populationSize() const override { return pop_.size(); }
    const std::vector<float>& weights(size_t i) const override { return pop_[i]; }
    void setFitness(size_t i, float f) override { fitness_[i] = f; }
    void evolve() override;
    int  generation() const override { return gen_; }
    const char* name() const override { return "random_search"; }
private:
    std::vector<std::vector<float>> pop_;
    std::vector<float>              fitness_;
    std::vector<float>              bestW_;
    float                           bestF_ = -std::numeric_limits<float>::infinity();
    int                             gen_   = 0;
    unsigned                        seed_  = 0;
    size_t                          wc_    = 0;

    static std::uint32_t lcg(std::uint32_t& s);
    static float         lcgFloat(std::uint32_t& s);
    void fillRandom(std::vector<float>& w, std::uint32_t& rng);
};

class HillClimbTrainer : public Trainer {
public:
    void init(size_t popSize, size_t weightCount, unsigned seed) override;
    void initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) override;
    size_t populationSize() const override { return pop_.size(); }
    const std::vector<float>& weights(size_t i) const override { return pop_[i]; }
    void setFitness(size_t i, float f) override { fitness_[i] = f; }
    void evolve() override;
    int  generation() const override { return gen_; }
    const char* name() const override { return "hillclimb"; }
private:
    std::vector<std::vector<float>> pop_;
    std::vector<float>              fitness_;
    std::vector<float>              cur_;
    float                           curF_  = -std::numeric_limits<float>::infinity();
    float                           sigma_ = 0.3f;
    int                             gen_   = 0;
    unsigned                        seed_  = 0;
    size_t                          wc_    = 0;

    static std::uint32_t lcg(std::uint32_t& s);
    static float         lcgFloat(std::uint32_t& s);
    float gaussNoise(std::uint32_t& rng);
    void rebuild();
};

std::unique_ptr<Trainer> makeTrainer(const std::string& algo);
