#include "Game.h"
#include "NeuralNetwork.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>

Game::Game(SimConfig cfg)
    : cfg_(std::move(cfg))
    , track_(std::make_unique<Track>(cfg_.map))
{
    numThreads_ = cfg_.threads > 0
        ? cfg_.threads
        : (int)std::thread::hardware_concurrency();
    if (numThreads_ < 1) numThreads_ = 1;

    cars_.resize(cfg_.population);
    // Default controllers: all NN with random weights
    controllers_.reserve(cfg_.population);
    std::mt19937 rng(cfg_.seed);
    for (int i = 0; i < cfg_.population; ++i) {
        unsigned s = (unsigned)rng();
        controllers_.push_back(
            std::make_unique<NeuralNetworkController>(NeuralNetwork(defaultTopology(), s)));
    }
    spawnCars();
}

void Game::spawnCars() {
    Vec2  sp = track_->spawnPos();
    float sa = track_->spawnAngle();
    for (auto& car : cars_) {
        car.reset(sp, sa);
        // Small random offset so cars don't stack exactly (debug aid)
    }
    for (auto& ctrl : controllers_) ctrl->reset();
}

void Game::setControllers(std::vector<std::unique_ptr<AIController>> ctrls) {
    if ((int)ctrls.size() != cfg_.population)
        throw std::runtime_error("Game::setControllers: size mismatch");
    controllers_ = std::move(ctrls);
}

void Game::loadMap(const std::string& path) {
    track_ = std::make_unique<Track>(path);
    cfg_.map = path;
    spawnCars();
}

void Game::updateRange(int begin, int end) {
    for (int i = begin; i < end; ++i) {
        Car& car = cars_[i];
        if (car.done) continue;
        Observation obs  = car.observe(*track_);
        Action      act  = controllers_[i]->decide(obs);
        car.applyAction(act);
        car.stepDone(*track_, cfg_.reward);
    }
}

void Game::tick() {
    int n = cfg_.population;
    if (numThreads_ <= 1 || n < 64) {
        updateRange(0, n);
    } else {
        int chunk = (n + numThreads_ - 1) / numThreads_;
        std::vector<std::thread> threads;
        threads.reserve(numThreads_);
        for (int t = 0; t < numThreads_; ++t) {
            int b = t * chunk;
            int e = std::min(b + chunk, n);
            if (b >= e) break;
            threads.emplace_back([this, b, e]{ updateRange(b, e); });
        }
        for (auto& th : threads) th.join();
    }

    // Force-kill stragglers: when <= 2% (min 2) of population remains alive,
    // they are not worth waiting for and would stall the generation.
    int straggler_threshold = std::max(2, n / 50);
    int alive = 0;
    for (const auto& c : cars_) if (!c.done) ++alive;
    if (alive > 0 && alive <= straggler_threshold) {
        for (auto& c : cars_)
            if (!c.done) { c.done = true; c.doneReason = DoneReason::Timeout; }
    }
}

bool Game::episodeDone() const {
    for (const auto& c : cars_)
        if (!c.done) return false;
    return true;
}

std::vector<float> Game::fitnesses() const {
    std::vector<float> f;
    f.reserve(cars_.size());
    for (const auto& c : cars_) f.push_back(c.fitness);
    return f;
}

// Single-agent RL interface
Observation Game::reset() {
    spawnCars();
    cars_[0].sensor.update(cars_[0].pos, cars_[0].angle, *track_);
    return cars_[0].observe(*track_);
}

StepResult Game::step(const Action& a) {
    Car& car = cars_[0];
    car.applyAction(a);
    float r = car.stepDone(*track_, cfg_.reward);
    Observation obs = car.observe(*track_);
    return {obs, r, car.done};
}

double Game::runHeadlessEpisode() {
    spawnCars();
    auto t0 = std::chrono::high_resolution_clock::now();
    while (!episodeDone()) tick();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void Game::runBenchmark(const SimConfig& baseCfg) {
    SimConfig cfg = baseCfg;
    cfg.headless = true;

    std::cout << "Benchmark: population=" << cfg.population
              << " threads=" << (cfg.threads > 0 ? cfg.threads : (int)std::thread::hardware_concurrency())
              << "\n";

    Game g(cfg);
    double secs = g.runHeadlessEpisode();

    int ticks = 0;
    for (const auto& c : g.cars()) ticks = std::max(ticks, (int)(c.episodeTime * SIM_HZ));
    std::cout << "  Wall-clock: " << secs << "s"
              << "  (~" << (int)(cfg.population * SIM_HZ * EPISODE_TIMEOUT) << " car-ticks total)\n";
}
