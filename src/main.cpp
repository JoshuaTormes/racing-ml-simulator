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
#include <sstream>
#include <stdexcept>

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

static std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(tok);
    return out;
}

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "\nBasic\n"
        << "  --headless            Run without window\n"
        << "  --map <path>          Map JSON for watch/single-map train (default: maps/map1.json)\n"
        << "  --population <N>      Number of cars (default: 1)\n"
        << "  --seed <S>            RNG seed (default: 42)\n"
        << "  --threads <K>         Thread count (default: hardware_concurrency)\n"
        << "  --benchmark           Run benchmark and exit\n"
        << "\nTraining\n"
        << "  --train               Run generational training loop\n"
        << "  --algo <name>         Training algorithm: genetic|random_search|hillclimb (default: genetic)\n"
        << "  --generations <N>     Number of generations (default: 100)\n"
        << "  --out <dir>           Output directory for weights (default: out/)\n"
        << "  --load <file.rnnw>    With --train: seed population from champion. Without: watch the network\n"
        << "  --watch <file.rnnw>   Watch a saved network drive (no training)\n"
        << "  --versus <file.rnnw>  Human (↑↓←→ / WASD) vs saved network. Yellow = you, green = AI\n"
        << "\nMulti-map training & robustness\n"
        << "  --train-maps <a,b,..> Comma-separated list of training maps\n"
        << "  --val-maps <x,y>      Comma-separated held-out maps (not used in selection)\n"
        << "  --fitness-agg <mode>  Aggregation: cvar-rank (default) | min | mean | cvar-raw\n"
        << "  --cvar-alpha <α>      CVaR tail fraction in (0,1] (default: 0.5). Used with cvar-rank/cvar-raw\n"
        << "  --map-norm <mode>     Per-map normalisation: zscore (default) | minmax | progress\n"
        << "                        Ignored under cvar-rank (ranks are scale-invariant)\n"
        << "  --curriculum <mode>   Map curriculum: linear (default) | none | explicit\n"
        << "  --curriculum-start <k>  Initial active map count for linear (default: 2)\n"
        << "  --curriculum-step <g>   Gens between additions for linear (default: 15)\n"
        << "  --curriculum-schedule <g1,g2,...>  Gen thresholds for explicit (M-1 values)\n"
        << "  --hidden <N>          Hidden layer size override (default: 32)\n"
        << "  --log-csv             Write training_log.csv and held_out_log.csv to --out dir\n"
        << "\n  Default when --train is active and no --train-maps is given:\n"
        << "    all *.json in maps/, sorted; first 6 = train, last 2 = val (reproducible split).\n"
        << "  Single-map compat: --map without --train-maps trains only on that map.\n"
        << "\n  Baseline (old behaviour) reproducible with:\n"
        << "    --fitness-agg min --map-norm progress --curriculum none\n";
}

static std::vector<float> loadChampion(const std::string& path) {
    NeuralNetwork nn(defaultTopology());
    nn.load(path);
    return nn.getWeights();
}

static FitnessAgg parseAgg(const std::string& s) {
    if (s == "cvar-rank") return FitnessAgg::CVaRRank;
    if (s == "min")       return FitnessAgg::Min;
    if (s == "mean")      return FitnessAgg::Mean;
    if (s == "cvar-raw")  return FitnessAgg::CVaRRaw;
    throw std::runtime_error("Invalid --fitness-agg value: " + s +
                             ". Valid: cvar-rank | min | mean | cvar-raw");
}

static MapNormMode parseNorm(const std::string& s) {
    if (s == "zscore")   return MapNormMode::ZScore;
    if (s == "minmax")   return MapNormMode::MinMax;
    if (s == "progress") return MapNormMode::Progress;
    throw std::runtime_error("Invalid --map-norm value: " + s +
                             ". Valid: zscore | minmax | progress");
}

static CurriculumMode parseCurriculum(const std::string& s) {
    if (s == "none")     return CurriculumMode::None;
    if (s == "linear")   return CurriculumMode::Linear;
    if (s == "explicit") return CurriculumMode::Explicit;
    throw std::runtime_error("Invalid --curriculum value: " + s +
                             ". Valid: none | linear | explicit");
}

int main(int argc, char* argv[]) {
    SimConfig cfg;
    bool benchmark    = false;
    bool train        = false;
    bool showHelp     = false;
    std::string algo        = "genetic";
    int         generations = 100;
    std::string outDir      = "out/";
    std::string loadPath;
    std::string watchPath;
    std::string versusPath;
    std::string trainMapsArg;
    std::string valMapsArg;
    bool explicitMap        = false;
    bool explicitTrainMaps  = false;

    // Multi-map robustness flags
    std::string fitnessAggArg   = "cvar-rank";
    float       cvarAlpha       = 0.5f;
    std::string mapNormArg      = "zscore";
    std::string curriculumArg   = "linear";
    int         currStart       = 2;
    int         currStep        = 15;
    std::string currScheduleArg;
    int         hiddenOverride  = -1;  // -1 = not specified
    bool        logCsv          = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            if      (arg == "--headless")                    cfg.headless    = true;
            else if (arg == "--benchmark")                   { benchmark = true; cfg.headless = true; }
            else if (arg == "--train")                       train           = true;
            else if (arg == "--map"          && i+1 < argc)  { cfg.map = argv[++i]; explicitMap = true; }
            else if (arg == "--population"   && i+1 < argc)  cfg.population  = std::stoi(argv[++i]);
            else if (arg == "--seed"         && i+1 < argc)  cfg.seed        = (unsigned)std::stoul(argv[++i]);
            else if (arg == "--threads"      && i+1 < argc)  cfg.threads     = std::stoi(argv[++i]);
            else if (arg == "--algo"         && i+1 < argc)  algo            = argv[++i];
            else if (arg == "--generations"  && i+1 < argc)  generations     = std::stoi(argv[++i]);
            else if (arg == "--out"          && i+1 < argc)  outDir          = argv[++i];
            else if (arg == "--load"         && i+1 < argc)  loadPath        = argv[++i];
            else if (arg == "--watch"        && i+1 < argc)  watchPath       = argv[++i];
            else if (arg == "--versus"       && i+1 < argc)  versusPath      = argv[++i];
            else if (arg == "--train-maps"   && i+1 < argc)  { trainMapsArg = argv[++i]; explicitTrainMaps = true; }
            else if (arg == "--val-maps"     && i+1 < argc)  valMapsArg      = argv[++i];
            else if (arg == "--fitness-agg"  && i+1 < argc)  fitnessAggArg   = argv[++i];
            else if (arg == "--cvar-alpha"   && i+1 < argc)  cvarAlpha       = std::stof(argv[++i]);
            else if (arg == "--map-norm"     && i+1 < argc)  mapNormArg      = argv[++i];
            else if (arg == "--curriculum"   && i+1 < argc)  curriculumArg   = argv[++i];
            else if (arg == "--curriculum-start" && i+1 < argc)  currStart   = std::stoi(argv[++i]);
            else if (arg == "--curriculum-step"  && i+1 < argc)  currStep    = std::stoi(argv[++i]);
            else if (arg == "--curriculum-schedule" && i+1 < argc) currScheduleArg = argv[++i];
            else if (arg == "--hidden"       && i+1 < argc)  hiddenOverride  = std::stoi(argv[++i]);
            else if (arg == "--log-csv")                      logCsv         = true;
            else if (arg == "--help" || arg == "-h")          showHelp       = true;
            else { std::cerr << "Unknown option: " << arg << "\n"; showHelp = true; }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing " << arg << ": " << e.what() << "\n";
            return 1;
        }
    }

    if (showHelp) { printUsage(argv[0]); return 0; }

    // ---- Validate and parse multi-map config --------------------------------
    FitnessAgg   fitnessAgg = FitnessAgg::CVaRRank;
    MapNormMode  mapNorm    = MapNormMode::ZScore;
    CurriculumMode currMode = CurriculumMode::Linear;

    try {
        fitnessAgg = parseAgg(fitnessAggArg);
        mapNorm    = parseNorm(mapNormArg);
        currMode   = parseCurriculum(curriculumArg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (cvarAlpha <= 0.f || cvarAlpha > 1.f) {
        std::cerr << "--cvar-alpha must be in (0, 1], got " << cvarAlpha << "\n";
        return 1;
    }

    // Warn if map-norm non-default but will be ignored under cvar-rank
    if (fitnessAgg == FitnessAgg::CVaRRank && mapNorm != MapNormMode::ZScore)
        std::cerr << "[warn] --map-norm is ignored under cvar-rank aggregation\n";

    // ---- Resolve --hidden and --load ----------------------------------------
    // Must be done before any NeuralNetwork construction.
    if (!loadPath.empty()) {
        // Read topology from saved file to detect hidden size
        std::vector<int> topo;
        try {
            topo = readTopologyFromFile(loadPath);
        } catch (const std::exception& e) {
            std::cerr << "Cannot read " << loadPath << ": " << e.what() << "\n";
            return 1;
        }
        int fileH = (topo.size() >= 3) ? topo[1] : -1;

        if (hiddenOverride >= 0) {
            if (hiddenOverride != fileH) {
                std::cerr << "--hidden " << hiddenOverride
                          << " conflicts with champion hidden size " << fileH
                          << " in " << loadPath << "\n";
                return 1;
            }
            setHiddenSize(hiddenOverride);
        } else {
            // Auto-detect from file topology
            if (fileH <= 0) {
                std::cerr << "Cannot infer hidden size from " << loadPath
                          << " (topology has " << topo.size() << " layers)\n";
                return 1;
            }
            setHiddenSize(fileH);
        }
    } else if (hiddenOverride >= 0) {
        setHiddenSize(hiddenOverride);
    }

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
            NeuralNetwork nn(defaultTopology());
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

    // ---- versus ----
    auto runVersus = [&](const std::string& path) {
#ifndef HEADLESS_ONLY
        std::vector<float> w = loadChampion(path);
        SimConfig vcfg = cfg;
        vcfg.population = 2;
        Game game(vcfg);
        {
            NeuralNetwork nn(defaultTopology());
            nn.setWeights(w);
            std::vector<std::unique_ptr<AIController>> ctrls;
            ctrls.push_back(std::make_unique<HumanController>());
            ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
            game.setControllers(std::move(ctrls));
        }
        Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
        game.reset();
        {
            auto mapDir = std::filesystem::path(vcfg.map).parent_path().string();
            auto maps = listMaps(mapDir);
            int idx = findMapIndex(maps, vcfg.map);
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
        std::cerr << "--versus requires a windowed build (no HEADLESS_ONLY)\n";
#endif
    };

    if (!watchPath.empty()) { runWatch(watchPath); return 0; }
    if (!versusPath.empty()) { runVersus(versusPath); return 0; }
    if (!loadPath.empty() && !train) { runWatch(loadPath); return 0; }

    // ---- training ----
    if (train) {
        // ----- Resolve train/val map lists -----------------------------------
        std::vector<std::string> trainMaps;
        std::vector<std::string> valMaps;

        if (explicitTrainMaps) {
            trainMaps = splitComma(trainMapsArg);
            if (!valMapsArg.empty())
                valMaps = splitComma(valMapsArg);
        } else if (explicitMap) {
            trainMaps = { cfg.map };
            if (!valMapsArg.empty())
                valMaps = splitComma(valMapsArg);
        } else {
            auto allMaps = listMaps("maps");
            if (allMaps.size() >= 3) {
                size_t nVal   = std::min((size_t)2, allMaps.size() / 4);
                size_t nTrain = allMaps.size() - nVal;
                trainMaps.assign(allMaps.begin(), allMaps.begin() + (long)nTrain);
                valMaps.assign(allMaps.begin() + (long)nTrain, allMaps.end());
            } else {
                trainMaps = allMaps;
            }
            if (!valMapsArg.empty())
                valMaps = splitComma(valMapsArg);
        }

        if (trainMaps.empty()) {
            std::cerr << "No training maps found. Check maps/ directory or --train-maps.\n";
            return 1;
        }

        // ----- Validate curriculum-schedule ----------------------------------
        CurriculumConfig currCfg;
        currCfg.mode  = currMode;
        currCfg.start = currStart;
        currCfg.step  = currStep;

        if (currMode == CurriculumMode::Explicit) {
            if (currScheduleArg.empty()) {
                std::cerr << "--curriculum explicit requires --curriculum-schedule <g1,g2,...>\n";
                return 1;
            }
            auto tokens = splitComma(currScheduleArg);
            int expected = (int)trainMaps.size() - 1;
            if ((int)tokens.size() != expected) {
                std::cerr << "--curriculum-schedule must have exactly M-1=" << expected
                          << " values for " << trainMaps.size() << " train maps; got "
                          << tokens.size() << "\n";
                return 1;
            }
            for (const auto& tok : tokens)
                currCfg.schedule.push_back(std::stoi(tok));
        }

        // ----- Build MultiMapConfig ------------------------------------------
        MultiMapConfig mmCfg;
        mmCfg.fitnessAgg  = fitnessAgg;
        mmCfg.cvarAlpha   = cvarAlpha;
        mmCfg.mapNorm     = mapNorm;
        mmCfg.curriculum  = currCfg;

        // ----- Print config summary ------------------------------------------
        std::string aggName = fitnessAggArg;
        if (fitnessAgg == FitnessAgg::CVaRRank || fitnessAgg == FitnessAgg::CVaRRaw)
            aggName += " α=" + std::to_string(cvarAlpha);
        std::cout << "Training: " << trainMaps.size() << " train map(s)";
        if (!valMaps.empty()) std::cout << ", " << valMaps.size() << " val map(s)";
        std::cout << " | agg=" << aggName
                  << " | curriculum=" << curriculumArg
                  << " | pop=" << cfg.population
                  << " | gen=" << generations << "\n";
        for (size_t i = 0; i < trainMaps.size(); ++i)
            std::cout << "  train[" << i << "]: " << trainMaps[i] << "\n";
        for (size_t i = 0; i < valMaps.size(); ++i)
            std::cout << "  val["   << i << "]: " << valMaps[i]   << "\n";

        // ----- Champion seed -------------------------------------------------
        const std::vector<float>* champion = nullptr;
        std::vector<float> championWeights;
        if (!loadPath.empty()) {
            championWeights = loadChampion(loadPath);
            champion = &championWeights;
        }

        if (cfg.headless) {
            TrainingSession session(cfg, makeTrainer(algo), generations, outDir,
                                    trainMaps, valMaps, mmCfg, champion, logCsv);
            session.runAll();
            return 0;
        }

#ifndef HEADLESS_ONLY
        bool multiMap = (trainMaps.size() > 1);
        TrainingSession session(cfg, makeTrainer(algo), generations, outDir,
                                trainMaps, valMaps, mmCfg, champion, logCsv);
        Renderer renderer(900, 700, "assets/DejaVuSans.ttf");

        {
        auto mapDir = std::filesystem::path(cfg.map).parent_path().string();
        auto maps = listMaps(mapDir);
        int idx = findMapIndex(maps, cfg.map);

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

    // ---- headless default ----
    if (cfg.headless) {
        Game game(cfg);
        double secs = game.runHeadlessEpisode();
        std::cout << "Headless episode done in " << secs << "s\n";
        return 0;
    }

    // ---- windowed default ----
#ifndef HEADLESS_ONLY
    Game game(cfg);
    {
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<HumanController>());
        std::mt19937 rng(cfg.seed + 1);
        for (int i = 1; i < cfg.population; ++i)
            ctrls.push_back(std::make_unique<NeuralNetworkController>(
                NeuralNetwork(defaultTopology(), (unsigned)rng())));
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
    }
#endif

    return 0;
}
