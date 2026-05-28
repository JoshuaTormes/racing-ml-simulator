#pragma once
#include "Trainer.h"
#include "Game.h"
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>

enum class FitnessAgg  { CVaRRank, Min, Mean, CVaRRaw };
enum class MapNormMode { ZScore, MinMax, Progress };
enum class CurriculumMode { None, Linear, Explicit };

struct CurriculumConfig {
    CurriculumMode   mode     = CurriculumMode::Linear;
    int              start    = 2;
    int              step     = 15;
    std::vector<int> schedule; // gen thresholds for mode==Explicit (M-1 values)
};

struct MultiMapConfig {
    FitnessAgg     fitnessAgg  = FitnessAgg::CVaRRank;
    float          cvarAlpha   = 0.5f;
    MapNormMode    mapNorm     = MapNormMode::ZScore;
    CurriculumConfig curriculum;
};

class TrainingSession {
public:
    TrainingSession(SimConfig sim,
                    std::unique_ptr<Trainer> trainer,
                    int generations,
                    std::string outDir,
                    std::vector<std::string> trainMaps       = {},
                    std::vector<std::string> valMaps         = {},
                    MultiMapConfig           mmCfg           = {},
                    const std::vector<float>* resumeChampion = nullptr,
                    bool                     logCsv          = false);

    void beginGeneration();
    void tick();
    bool generationComplete() const;
    void endGeneration();
    bool done() const;

    void evaluateGenerationParallel();
    void finalizeGeneration();

    void setMap(const std::string& path);
    void runAll();

    const Game&  game()              const { return game_; }
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

    std::vector<std::string>        trainMaps_;
    std::vector<std::string>        valMaps_;
    MultiMapConfig                  mmCfg_;
    int                             currentMapInGen_ = 0;
    std::vector<std::vector<float>> perMapFitness_;
    int                             valEvery_ = 10;

    std::vector<std::unique_ptr<Track>> trainTracks_;
    std::vector<std::unique_ptr<Track>> valTracks_;

    int workerCount_ = 1;

    std::vector<DoneReason>              diagReasons_;
    std::vector<std::vector<DoneReason>> perMapDoneReasons_;

    bool          logCsv_ = false;
    std::ofstream trainingLog_;
    std::ofstream heldOutLog_;

    void loadTrainMap(size_t mapIdx);
    void recordCurrentMapFitness();
    std::vector<float> aggregateFitness(int activeMaps) const;
    void evaluateHeldOut(size_t championIdx);
    void advanceIfMapDone();

    void printStats(const GenerationStats& s) const;
    void writeTrainingCsvHeader();
    void writeTrainingCsvRow(const GenerationStats& s);
};
