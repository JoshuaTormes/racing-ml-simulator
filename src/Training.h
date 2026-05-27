#pragma once
#include "Trainer.h"
#include "Game.h"
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

enum class FitnessAgg { Min, Mean };

class TrainingSession {
public:
    TrainingSession(SimConfig sim,
                    std::unique_ptr<Trainer> trainer,
                    int generations,
                    std::string outDir,
                    std::vector<std::string> trainMaps = {},
                    std::vector<std::string> valMaps   = {},
                    FitnessAgg agg                     = FitnessAgg::Min,
                    const std::vector<float>* resumeChampion = nullptr);

    void beginGeneration();
    void tick();
    bool generationComplete() const;
    void endGeneration();
    bool done() const;

    // Headless parallel evaluation (fills perMapFitness_ for all maps × cars)
    void evaluateGenerationParallel();
    // Common finalization: aggregate + stats + saves + held-out + evolve
    void finalizeGeneration();

    // Replace active track; in single-map mode updates trainMaps_[0] and restarts.
    // In multi-map mode restarts the generation from trainMaps_[0] (path ignored).
    void setMap(const std::string& path);

    void runAll();

    const Game&  game()             const { return game_; }
    int          currentGeneration() const { return currentGen_; }
    int          totalGenerations()  const { return generations_; }
    const char*  algoName()          const { return trainer_->name(); }
    const GenerationStats& lastStats() const { return lastStats_; }
    const std::vector<GenerationStats>& history() const { return history_; }

private:
    Game                         game_;
    std::unique_ptr<Trainer>     trainer_;
    int                          generations_;
    std::string                  outDir_;
    int                          currentGen_ = 0;
    float                        bestGlobalF_;
    GenerationStats              lastStats_;
    std::vector<GenerationStats> history_;

    // Multi-map members
    std::vector<std::string>        trainMaps_;
    std::vector<std::string>        valMaps_;
    FitnessAgg                      agg_;
    int                             currentMapInGen_ = 0;
    std::vector<std::vector<float>> perMapFitness_;  // [mapIdx][carIdx] raw fitness
    std::vector<float>              mapNorm_;         // normalisation constant per train map
    int                             valEvery_ = 10;  // held-out evaluation frequency (gens)

    // Cached tracks (loaded once in constructor; reused every generation)
    std::vector<std::unique_ptr<Track>> trainTracks_;
    std::vector<std::unique_ptr<Track>> valTracks_;

    // Parallelism
    int                          workerCount_ = 1;

    // Per-car done-reason from diagnostic map (map 0), shared between paths
    std::vector<DoneReason>      diagReasons_;

    void loadTrainMap(size_t mapIdx);
    void recordCurrentMapFitness();
    std::vector<float> aggregateFitness() const;
    void evaluateHeldOut(size_t championIdx);
    float mapNormConst(const Track& t) const;
    void advanceIfMapDone();

    void printStats(const GenerationStats& s) const;
};
