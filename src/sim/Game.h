#pragma once
#include "core/Types.h"
#include "sim/Track.h"
#include "sim/Car.h"
#include "control/AIController.h"
#include <vector>
#include <memory>
#include <thread>
#include <functional>

// Per-episode randomization for Game::simulateEpisode. Defaults reproduce the legacy
// deterministic episode exactly (fixed spawn, no sensor noise) — the seed is only
// consulted when randomSpawn or sensorNoise>0 is set.
struct EpisodeConfig {
    unsigned seed        = 0;
    bool     randomSpawn = false;
    float    sensorNoise = 0.f;   // stddev of gaussian noise added to ray readings
};

class Game {
public:
    explicit Game(SimConfig cfg);

    // Single-agent RL interface (population == 1)
    Observation reset();
    StepResult  step(const Action& a);

    // Batch (neuroevolution): advance all cars one tick
    void tick();
    bool episodeDone() const;

    // Collect fitness per car after episode
    std::vector<float> fitnesses() const;

    // Run full headless episode; returns wall-clock seconds
    double runHeadlessEpisode();

    // Benchmark: N cars, random NN controllers, 1 episode; prints timing
    static void runBenchmark(const SimConfig& cfg);

    // Access cars/track for Renderer
    const std::vector<Car>& cars()  const { return cars_; }
    const Track&            track() const { return *track_; }

    // Set controllers (takes ownership); must match population size
    void setControllers(std::vector<std::unique_ptr<AIController>> ctrls);

    // Replace the active track and respawn all cars (SFML-free)
    void loadMap(const std::string& path);
    void loadMap(const Track& t);  // in-memory variant (no file I/O)

    const SimConfig& config() const { return cfg_; }

    // ---------------------------------------------------------------------------
    // Simulate a single car (ctrl) on a track to completion; no Game state used.
    // Thread-safe: only reads track/reward; ctrl and Car are local to the caller.
    // ---------------------------------------------------------------------------
    struct EpisodeResult {
        float      fitness     = 0.f;
        float      maxProgress = 0.f;
        float      episodeTime = 0.f;
        DoneReason doneReason  = DoneReason::None;
    };
    static EpisodeResult simulateEpisode(const Track& track,
                                         AIController& ctrl,
                                         const RewardConfig& reward,
                                         const EpisodeConfig& ep = {});

private:
    SimConfig                              cfg_;
    std::unique_ptr<Track>                 track_;
    std::vector<Car>                       cars_;
    std::vector<std::unique_ptr<AIController>> controllers_;
    int                                    numThreads_;

    void spawnCars();
    // Update a range [begin, end) of cars in the current tick
    void updateRange(int begin, int end);
};
