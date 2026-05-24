#pragma once
#include "Trainer.h"
#include "Game.h"
#include <memory>
#include <string>
#include <vector>

class TrainingSession {
public:
    TrainingSession(SimConfig sim,
                    std::unique_ptr<Trainer> trainer,
                    int generations,
                    std::string outDir,
                    const std::vector<float>* resumeChampion = nullptr);

    void beginGeneration();
    void tick();
    bool generationComplete() const;
    void endGeneration();
    bool done() const;

    // Replace active track and restart current generation (SFML-free)
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

    void printStats(const GenerationStats& s) const;
};
