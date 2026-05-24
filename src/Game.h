#pragma once
#include "core/Types.h"
#include "Track.h"
#include "Car.h"
#include "AIController.h"
#include <vector>
#include <memory>
#include <thread>
#include <functional>

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

    const SimConfig& config() const { return cfg_; }

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
