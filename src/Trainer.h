#pragma once
#include <vector>
#include <cstddef>

struct GenerationStats {
    int   generation  = 0;
    float bestFitness = 0.f;
    float meanFitness = 0.f;
    float stdFitness  = 0.f;
    int   completed   = 0;
    int   population  = 0;
    // Done reason breakdown
    int   nCollision  = 0;
    int   nStall      = 0;
    int   nTimeout    = 0;
};

class Trainer {
public:
    virtual ~Trainer() = default;
    virtual void init(size_t popSize, size_t weightCount, unsigned seed) = 0;
    virtual void initFromWeights(const std::vector<float>& champion,
                                 size_t popSize, unsigned seed) = 0;
    virtual size_t populationSize() const = 0;
    virtual const std::vector<float>& weights(size_t i) const = 0;
    virtual void setFitness(size_t i, float f) = 0;
    virtual void evolve() = 0;
    virtual int  generation() const = 0;
    virtual const char* name() const = 0;
};
