#include "app/Runners.h"
#include "cli/Util.h"
#include "control/NeuralNetwork.h"
#include "core/Constants.h"
#include "sim/Game.h"
#include "training/Trainers.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#ifndef HEADLESS_ONLY
#include "control/HumanController.h"
#include "render/InteractiveLoop.h"
#include "render/Renderer.h"
#include <chrono>
#include <filesystem>
#endif

int runBenchmark(const AppConfig& cfg) {
    Game::runBenchmark(cfg.sim);
    return 0;
}

int runWatch(const AppConfig& cfg) {
#ifndef HEADLESS_ONLY
    std::vector<float> w = loadChampion(cfg.watchSourcePath);
    SimConfig wcfg = cfg.sim;
    wcfg.population = 1;
    Game game(wcfg);
    {
        NeuralNetwork nn(defaultTopology());
        nn.setWeights(w);
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
        game.setControllers(std::move(ctrls));
    }
    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
    runInteractiveLoop(game, renderer, wcfg.map);
#else
    std::cerr << "--watch requires a windowed build (no HEADLESS_ONLY)\n";
#endif
    return 0;
}

int runVersus(const AppConfig& cfg) {
#ifndef HEADLESS_ONLY
    std::vector<float> w = loadChampion(cfg.versusPath);
    SimConfig vcfg = cfg.sim;
    int nAI = std::max(1, cfg.sim.population);
    vcfg.population = 1 + nAI;
    Game game(vcfg);
    {
        std::mt19937 rng(vcfg.seed);
        std::normal_distribution<float> noise(0.f, cfg.versusNoise);
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<HumanController>());
        for (int i = 0; i < nAI; ++i) {
            NeuralNetwork nn(defaultTopology());
            std::vector<float> wn = w;
            if (cfg.versusNoise > 0.f)
                for (float& x : wn) x += noise(rng);
            nn.setWeights(wn);
            ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
        }
        game.setControllers(std::move(ctrls));
    }
    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
    runInteractiveLoop(game, renderer, vcfg.map);
#else
    std::cerr << "--versus requires a windowed build (no HEADLESS_ONLY)\n";
#endif
    return 0;
}

int runInteractive(const AppConfig& cfg) {
#ifndef HEADLESS_ONLY
    Game game(cfg.sim);
    {
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<HumanController>());
        std::mt19937 rng(cfg.sim.seed + 1);
        for (int i = 1; i < cfg.sim.population; ++i)
            ctrls.push_back(std::make_unique<NeuralNetworkController>(
                NeuralNetwork(defaultTopology(), (unsigned)rng())));
        game.setControllers(std::move(ctrls));
    }
    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
    runInteractiveLoop(game, renderer, cfg.sim.map);
#endif
    return 0;
}

int runHeadlessDefault(const AppConfig& cfg) {
    Game game(cfg.sim);
    double secs = game.runHeadlessEpisode();
    std::cout << "Headless episode done in " << secs << "s\n";
    return 0;
}

int runTraining(const AppConfig& cfg) {
    // ----- Print config summary ------------------------------------------
    std::string aggName = cfg.fitnessAggArg;
    if (cfg.mmCfg.fitnessAgg == FitnessAgg::CVaRRank || cfg.mmCfg.fitnessAgg == FitnessAgg::CVaRRaw)
        aggName += " α=" + std::to_string(cfg.mmCfg.cvarAlpha);
    std::cout << "Training: " << cfg.trainMaps.size() << " train map(s)";
    if (!cfg.valMaps.empty()) std::cout << ", " << cfg.valMaps.size() << " val map(s)";
    std::cout << " | agg=" << aggName
              << " | curriculum=" << cfg.curriculumArg
              << " | pop=" << cfg.sim.population
              << " | gen=" << cfg.generations << "\n";
    for (size_t i = 0; i < cfg.trainMaps.size(); ++i)
        std::cout << "  train[" << i << "]: " << cfg.trainMaps[i] << "\n";
    for (size_t i = 0; i < cfg.valMaps.size(); ++i)
        std::cout << "  val["   << i << "]: " << cfg.valMaps[i]   << "\n";

    const std::vector<float>* champion = cfg.hasChampion ? &cfg.championWeights : nullptr;

    if (cfg.sim.headless) {
        TrainingSession session(cfg.sim, makeTrainer(cfg.algo), cfg.generations, cfg.outDir,
                                cfg.trainMaps, cfg.valMaps, cfg.mmCfg, champion, cfg.logCsv, cfg.testMaps);
        session.runAll();
        return 0;
    }

#ifndef HEADLESS_ONLY
    bool multiMap = (cfg.trainMaps.size() > 1);
    TrainingSession session(cfg.sim, makeTrainer(cfg.algo), cfg.generations, cfg.outDir,
                            cfg.trainMaps, cfg.valMaps, cfg.mmCfg, champion, cfg.logCsv, cfg.testMaps);
    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");

    {
    auto mapDir = std::filesystem::path(cfg.sim.map).parent_path().string();
    auto maps = listMaps(mapDir);
    int idx = findMapIndex(maps, cfg.sim.map);

    session.beginGeneration();
    using Clock = std::chrono::steady_clock;
    auto prev = Clock::now();
    float accumulator = 0.f;

    const int MAX_TICKS_PER_FRAME = 500;

    while (renderer.isOpen() && !session.done()) {
        if (!renderer.handleEvents()) break;
        if (renderer.consumeRestart()) {
            session.beginGeneration();
            accumulator = 0.f;
            prev = Clock::now();
        }
        if (!multiMap) {
            if (int d = renderer.consumeMapDelta(); d && !maps.empty()) {
                idx = (idx + d + (int)maps.size()) % (int)maps.size();
                session.setMap(maps[idx]);
                accumulator = 0.f;
                prev = Clock::now();
            }
        } else {
            renderer.consumeMapDelta();
        }

        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        if (dt > 0.25f) dt = 0.25f;
        accumulator += dt * renderer.simSpeed();

        int budget = MAX_TICKS_PER_FRAME;
        while (accumulator >= DT && budget-- > 0) {
            if (session.generationComplete()) {
                session.endGeneration();
                if (session.done()) break;
                session.beginGeneration();
            }
            session.tick();
            accumulator -= DT;
        }
        if (accumulator > DT) accumulator = 0.f;

        renderer.renderTraining(session);
    }
    while (renderer.isOpen()) {
        if (!renderer.handleEvents()) break;
        renderer.renderTraining(session);
    }
    }
#endif
    return 0;
}

int dispatch(const AppConfig& cfg) {
    switch (cfg.mode) {
        case RunMode::Benchmark:       return runBenchmark(cfg);
        case RunMode::Watch:           return runWatch(cfg);
        case RunMode::Versus:          return runVersus(cfg);
        case RunMode::Train:           return runTraining(cfg);
        case RunMode::HeadlessDefault: return runHeadlessDefault(cfg);
        case RunMode::WindowedDefault: return runInteractive(cfg);
    }
    return 0;
}
