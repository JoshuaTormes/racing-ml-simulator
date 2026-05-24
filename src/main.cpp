#include "Game.h"
#include "HumanController.h"
#include "NeuralNetwork.h"
#include "Renderer.h"
#include "Trainers.h"
#include "Training.h"
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>

static std::vector<std::string> listMaps(const std::string& dir) {
    std::vector<std::string> v;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir.empty() ? "maps" : dir, ec))
        if (e.is_regular_file() && e.path().extension() == ".json")
            v.push_back(e.path().string());
    std::sort(v.begin(), v.end());
    return v;
}

static int findMapIndex(const std::vector<std::string>& maps, const std::string& cur) {
    auto curName = std::filesystem::path(cur).filename();
    for (size_t i = 0; i < maps.size(); ++i)
        if (std::filesystem::path(maps[i]).filename() == curName) return (int)i;
    return 0;
}

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --headless            Run without window\n"
        << "  --map <path>          Map JSON (default: maps/map1.json)\n"
        << "  --population <N>      Number of cars (default: 1)\n"
        << "  --seed <S>            RNG seed (default: 42)\n"
        << "  --threads <K>         Thread count (default: hardware_concurrency)\n"
        << "  --benchmark           Run benchmark and exit\n"
        << "  --train               Run generational training loop\n"
        << "  --algo <name>         Training algorithm: genetic|random_search|hillclimb (default: genetic)\n"
        << "  --generations <N>     Number of generations (default: 100)\n"
        << "  --out <dir>           Output directory for weights (default: out/)\n"
        << "  --load <file.rnnw>    With --train: seed population from champion. Without: watch the network\n"
        << "  --watch <file.rnnw>   Watch a saved network drive (no training)\n";
}

static std::vector<float> loadChampion(const std::string& path) {
    NeuralNetwork nn({OBS_SIZE, 8, 2});
    nn.load(path);
    return nn.getWeights();
}

int main(int argc, char* argv[]) {
    SimConfig cfg;
    bool benchmark   = false;
    bool train       = false;
    bool showHelp    = false;
    std::string algo        = "genetic";
    int         generations = 100;
    std::string outDir      = "out/";
    std::string loadPath;
    std::string watchPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--headless")                   cfg.headless    = true;
        else if (arg == "--benchmark")                  { benchmark = true; cfg.headless = true; }
        else if (arg == "--train")                      train           = true;
        else if (arg == "--map"         && i+1 < argc)  cfg.map         = argv[++i];
        else if (arg == "--population"  && i+1 < argc)  cfg.population  = std::stoi(argv[++i]);
        else if (arg == "--seed"        && i+1 < argc)  cfg.seed        = (unsigned)std::stoul(argv[++i]);
        else if (arg == "--threads"     && i+1 < argc)  cfg.threads     = std::stoi(argv[++i]);
        else if (arg == "--algo"        && i+1 < argc)  algo            = argv[++i];
        else if (arg == "--generations" && i+1 < argc)  generations     = std::stoi(argv[++i]);
        else if (arg == "--out"         && i+1 < argc)  outDir          = argv[++i];
        else if (arg == "--load"        && i+1 < argc)  loadPath        = argv[++i];
        else if (arg == "--watch"       && i+1 < argc)  watchPath       = argv[++i];
        else if (arg == "--help" || arg == "-h")        showHelp        = true;
        else { std::cerr << "Unknown option: " << arg << "\n"; showHelp = true; }
    }

    if (showHelp) { printUsage(argv[0]); return 0; }

    // ---- benchmark ----
    if (benchmark) {
        Game::runBenchmark(cfg);
        return 0;
    }

    // ---- watch / load-without-train ----
    auto runWatch = [&](const std::string& path) {
#ifndef HEADLESS_ONLY
        std::vector<float> w = loadChampion(path);
        SimConfig wcfg = cfg;
        wcfg.population = 1;
        Game game(wcfg);
        {
            NeuralNetwork nn({OBS_SIZE, 8, 2});
            nn.setWeights(w);
            std::vector<std::unique_ptr<AIController>> ctrls;
            ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
            game.setControllers(std::move(ctrls));
        }
        Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
        game.reset();
        {
            auto mapDir = std::filesystem::path(wcfg.map).parent_path().string();
            auto maps = listMaps(mapDir);
            int idx = findMapIndex(maps, wcfg.map);
            using Clock = std::chrono::steady_clock;
            auto prev = Clock::now();
            float accumulator = 0.f;
            while (renderer.isOpen()) {
                if (!renderer.handleEvents()) break;
                if (renderer.consumeRestart()) { game.reset(); accumulator = 0.f; }
                if (int d = renderer.consumeMapDelta(); d && !maps.empty()) {
                    idx = (idx + d + (int)maps.size()) % (int)maps.size();
                    game.loadMap(maps[idx]); accumulator = 0.f;
                }
                auto now = Clock::now();
                float dt = std::chrono::duration<float>(now - prev).count();
                prev = now;
                accumulator += dt;
                while (accumulator >= DT) {
                    game.tick();
                    accumulator -= DT;
                    if (game.episodeDone()) {
                        game.reset();
                        accumulator = 0.f;
                        break;
                    }
                }
                renderer.render(game, /*showRays=*/true);
            }
        }
#else
        (void)path;
        std::cerr << "--watch requires a windowed build (no HEADLESS_ONLY)\n";
#endif
    };

    if (!watchPath.empty()) { runWatch(watchPath); return 0; }
    if (!loadPath.empty() && !train) { runWatch(loadPath); return 0; }

    // ---- training ----
    if (train) {
        const std::vector<float>* champion = nullptr;
        std::vector<float> championWeights;
        if (!loadPath.empty()) {
            championWeights = loadChampion(loadPath);
            champion = &championWeights;
        }

        if (cfg.headless) {
            // Headless training
            TrainingSession session(cfg, makeTrainer(algo), generations, outDir, champion);
            session.runAll();
            return 0;
        }

#ifndef HEADLESS_ONLY
        // Windowed training
        TrainingSession session(cfg, makeTrainer(algo), generations, outDir, champion);
        Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
        bool turbo = false;

        {
        auto mapDir = std::filesystem::path(cfg.map).parent_path().string();
        auto maps = listMaps(mapDir);
        int idx = findMapIndex(maps, cfg.map);

        session.beginGeneration();
        using Clock = std::chrono::steady_clock;
        auto prev = Clock::now();
        float accumulator = 0.f;

        while (renderer.isOpen() && !session.done()) {
            if (!renderer.handleEvents()) break;
            if (renderer.consumeToggleTurbo()) turbo = !turbo;
            if (renderer.consumeRestart()) { session.beginGeneration(); accumulator = 0.f; prev = Clock::now(); }
            if (int d = renderer.consumeMapDelta(); d && !maps.empty()) {
                idx = (idx + d + (int)maps.size()) % (int)maps.size();
                session.setMap(maps[idx]); accumulator = 0.f; prev = Clock::now();
            }

            if (turbo) {
                // Run many ticks (up to full generation) per frame
                int budget = 2000;
                while (budget-- > 0 && !session.done()) {
                    if (session.generationComplete()) {
                        session.endGeneration();
                        if (!session.done()) session.beginGeneration();
                    } else {
                        session.tick();
                    }
                }
            } else {
                auto now = Clock::now();
                float dt = std::chrono::duration<float>(now - prev).count();
                prev = now;
                accumulator += dt;
                while (accumulator >= DT) {
                    if (!session.generationComplete())
                        session.tick();
                    accumulator -= DT;
                    if (session.generationComplete()) {
                        session.endGeneration();
                        if (!session.done()) session.beginGeneration();
                        accumulator = 0.f;
                        break;
                    }
                }
            }
            prev = Clock::now(); // keep prev in sync after turbo burst
            renderer.renderTraining(session, turbo);
        }
        // Keep window open showing final state
        while (renderer.isOpen()) {
            if (!renderer.handleEvents()) break;
            renderer.renderTraining(session, turbo);
        }
        } // maps scope
#endif
        return 0;
    }

    // ---- headless default ----
    if (cfg.headless) {
        Game game(cfg);
        double secs = game.runHeadlessEpisode();
        std::cout << "Headless episode done in " << secs << "s\n";
        return 0;
    }

    // ---- windowed default: car 0 by keyboard, rest by NN ----
#ifndef HEADLESS_ONLY
    Game game(cfg);
    {
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<HumanController>());
        std::mt19937 rng(cfg.seed + 1);
        for (int i = 1; i < cfg.population; ++i)
            ctrls.push_back(std::make_unique<NeuralNetworkController>(
                NeuralNetwork({OBS_SIZE, 8, 2}, (unsigned)rng())));
        game.setControllers(std::move(ctrls));
    }

    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
    game.reset();

    {
    auto mapDir = std::filesystem::path(cfg.map).parent_path().string();
    auto maps = listMaps(mapDir);
    int idx = findMapIndex(maps, cfg.map);

    using Clock = std::chrono::steady_clock;
    auto prev = Clock::now();
    float accumulator = 0.f;

    while (renderer.isOpen()) {
        if (!renderer.handleEvents()) break;
        if (renderer.consumeRestart()) { game.reset(); accumulator = 0.f; }
        if (int d = renderer.consumeMapDelta(); d && !maps.empty()) {
            idx = (idx + d + (int)maps.size()) % (int)maps.size();
            game.loadMap(maps[idx]); accumulator = 0.f;
        }

        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        accumulator += dt;

        while (accumulator >= DT) {
            game.tick();
            accumulator -= DT;
            if (game.episodeDone()) {
                game.reset();
                accumulator = 0.f;
                break;
            }
        }

        renderer.render(game, /*showRays=*/true);
    }
    } // maps scope
#endif

    return 0;
}
