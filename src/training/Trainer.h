#pragma once
#include <vector>
#include <cstddef>
#include <cmath>

struct PerMapStats {
    int   mapIndex    = -1;
    bool  active      = false;
    int   nEvaluated  = 0;
    float bestRaw     = 0.f;
    float meanRaw     = 0.f;
    float medianRaw   = 0.f;
    float stdRaw      = 0.f;
    float normalizedBest = 0.f;
    int   nCompleted  = 0;
    int   nCollision  = 0;
    int   nStall      = 0;
    int   nTimeout    = 0;
};

struct GenerationStats {
    int   generation  = 0;
    int   population  = 0;

    float aggBest      = 0.f;
    float aggMean      = 0.f;
    float aggMedian    = 0.f;
    float aggStd       = 0.f;
    float aggTopDecile = 0.f;
    float aggMin       = 0.f;

    std::vector<PerMapStats> perMap;
    int   activeMapCount = 0;
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
