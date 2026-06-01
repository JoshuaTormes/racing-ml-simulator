#pragma once
#include "training/Trainer.h"
#include "sim/Game.h"
#include "sim/TrackGen.h"
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>

enum class FitnessAgg  { CVaRRank, Min, Mean, CVaRRaw };
enum class MapNormMode { ZScore, MinMax, Progress };
enum class CurriculumMode { None, Linear, Explicit };
enum class EpisodeAgg  { Mean, Min }; // how to combine multiple episodes per (genome, map)

class NeuralNetwork; // fwd-decl for evalCar

struct CurriculumConfig {
    CurriculumMode   mode     = CurriculumMode::Linear;
    int              start    = 2;
    int              step     = 15;
    std::vector<int> schedule; // gen thresholds for mode==Explicit (M-1 values)
    std::vector<int> pinned;   // map indices always active regardless of curriculum
};

struct MultiMapConfig {
    FitnessAgg       fitnessAgg    = FitnessAgg::CVaRRank;
    float            cvarAlpha     = 0.5f;
    MapNormMode      mapNorm       = MapNormMode::ZScore;
    CurriculumConfig curriculum;
    std::vector<float> mapWeights; // per-map weights for CVaRRank; empty = all 1.0
    float            progressiveFrac = 1.0f; // fraction of population evaluated on maps[1..]

    // Held-out model selection: when true, best.rnnw is chosen by validation-map
    // performance (max mean progress among the top-K train genomes) instead of by
    // raw train fitness. Off by default → identical to legacy behavior.
    bool             selectByVal   = false;
    int              valSelectTopK = 1;

    // Per-episode anti-memorization randomization (training only; held-out/test stay
    // deterministic). Defaults reproduce the legacy single deterministic episode.
    int              episodesPerEval = 1;
    bool             randomSpawn     = false;
    float            sensorNoise     = 0.f;
    EpisodeAgg       episodeAgg      = EpisodeAgg::Mean;

    // Static augmentation tokens applied to each base train map: "mirror", "reverse",
    // "width:<factor>". Each produces an extra training map. Empty = no augmentation.
    std::vector<std::string> augment;

    // If non-empty, all generated/augmented train tracks are written as JSON files to
    // this directory before training starts (useful for visualization/debugging).
    std::string dumpGenMaps;

    // Procedural tracks generated (seeded from cfg.seed) and appended to the train/val
    // sets. 0 = none. genParams controls the generator (waypoint count, radius, width).
    int                 proceduralTrain = 0;
    int                 proceduralVal   = 0;
    trackgen::GenParams genParams;
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
                    bool                     logCsv          = false,
                    std::vector<std::string> testMaps        = {});

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
    std::vector<std::string>        testMaps_;
    MultiMapConfig                  mmCfg_;
    int                             currentMapInGen_ = 0;
    std::vector<std::vector<float>> perMapFitness_;
    int                             valEvery_ = 10;
    float                           bestValScore_; // high-water mean val progress (selectByVal)

    std::vector<std::unique_ptr<Track>> trainTracks_;
    std::vector<std::unique_ptr<Track>> valTracks_;
    std::vector<std::unique_ptr<Track>> testTracks_;

    int workerCount_ = 1;

    std::vector<DoneReason>              diagReasons_;
    std::vector<std::vector<DoneReason>> perMapDoneReasons_;

    bool          logCsv_ = false;
    std::ofstream trainingLog_;
    std::ofstream heldOutLog_;
    std::ofstream testLog_;

    // Evaluate one genome on one track over mmCfg_.episodesPerEval episodes (each with
    // per-episode randomization seeded deterministically), aggregated per episodeAgg.
    // With episodesPerEval==1 and no randomization this equals a single simulateEpisode.
    Game::EpisodeResult evalCar(const Track& track, const NeuralNetwork& net,
                                const RewardConfig& reward,
                                size_t mapIdx, size_t carIdx) const;

    void loadTrainMap(size_t mapIdx);
    void recordCurrentMapFitness();
    std::vector<float> aggregateFitness(const std::vector<int>& activeIdx) const;
    void evaluateHeldOut(size_t championIdx);
    // Simulate one champion on a set of tracks, printing/logging per-map progress.
    void logChampionOnMaps(const char* label,
                           const std::vector<std::unique_ptr<Track>>& tracks,
                           const std::vector<std::string>& names,
                           size_t championIdx,
                           std::ofstream& log);
    // Mean maxProgress of one weight vector over a set of tracks (no logging).
    float meanProgressOn(const std::vector<std::unique_ptr<Track>>& tracks,
                         const std::vector<float>& weights) const;
    // Pick best.rnnw by validation among the top-K train genomes; returns the
    // chosen champion index (for held-out/test logging). Updates bestValScore_.
    size_t selectChampionByVal(const std::vector<float>& agg);
    void advanceIfMapDone();

    void printStats(const GenerationStats& s) const;
    void writeTrainingCsvHeader();
    void writeTrainingCsvRow(const GenerationStats& s);
};
