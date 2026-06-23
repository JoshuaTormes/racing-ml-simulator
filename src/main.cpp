#include "sim/Game.h"
#include "control/HumanController.h"
#include "control/NeuralNetwork.h"
#include "render/Renderer.h"
#include "training/Trainers.h"
#include "training/Training.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Aggregates all raw CLI/JSON config state for main(). Populated by applyConfig()
// (from a config file) and then by the CLI parsing loop (which overrides it).
struct CliOptions {
    // Basic
    SimConfig   sim;
    bool        benchmark   = false;
    bool        showHelp    = false;
    bool        explicitMap = false;

    // Training
    bool        train             = false;
    std::string algo              = "genetic";
    int         generations       = 100;
    std::string outDir            = "out/";
    std::string loadPath;
    std::string watchPath;
    std::string versusPath;
    float       versusNoise       = 0.02f;
    bool        logCsv            = false;
    int         hiddenOverride    = -1;  // -1 = not specified
    float       episodeTimeoutArg = -1.f;

    // Maps
    std::string trainMapsArg;
    bool        explicitTrainMaps = false;
    std::string valMapsArg;
    std::string testMapsArg;

    // Fitness aggregation
    std::string fitnessAggArg   = "cvar-rank";
    float       cvarAlpha       = 0.5f;
    std::string mapNormArg      = "zscore";
    std::string mapWeightsArg;
    float       progressiveFrac = 1.0f;
    std::string finetuneMapArg;

    // Curriculum
    std::string curriculumArg   = "linear";
    int         currStart       = 2;
    int         currStep        = 15;
    std::string currScheduleArg;
    std::string curriculumPinArg;

    // Anti-overfitting
    std::string augmentArg;
    std::string dumpGenMapsDir;
    int         proceduralTrain  = 0;
    int         proceduralVal    = 0;
    float       procWidthMin     = -1.f;  // -1 = use generator default
    float       procWidthMax     = -1.f;
    bool        randomSpawn      = false;
    float       sensorNoise      = 0.f;
    int         episodesPerEval  = 1;
    std::string episodeAggArg    = "mean";

    // Model selection
    bool        selectByVal      = false;
    int         valSelectTopK    = 1;
};

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

// Joins a JSON string-array into a comma-separated string (same format the CLI expects).
static std::string jsonArrayToComma(const json& v) {
    std::string out;
    if (v.is_array()) {
        for (const auto& e : v) { if (!out.empty()) out += ","; out += e.get<std::string>(); }
    } else {
        out = v.get<std::string>();
    }
    return out;
}

// Apply values from a config JSON object to all CLI options.
// Called before the CLI parsing loop — subsequent CLI flags overwrite these values.
static void applyConfig(const json& j, CliOptions& opt) {
    // Basic
    if (j.contains("population"))    opt.sim.population    = j["population"].get<int>();
    if (j.contains("seed"))          opt.sim.seed           = j["seed"].get<unsigned>();
    if (j.contains("threads"))       opt.sim.threads        = j["threads"].get<int>();
    if (j.contains("headless"))      opt.sim.headless       = j["headless"].get<bool>();
    if (j.contains("map"))           opt.sim.map            = j["map"].get<std::string>();
    // Training
    if (j.contains("train"))         opt.train              = j["train"].get<bool>();
    if (j.contains("algo"))          opt.algo               = j["algo"].get<std::string>();
    if (j.contains("generations"))   opt.generations        = j["generations"].get<int>();
    if (j.contains("out"))           opt.outDir             = j["out"].get<std::string>();
    if (j.contains("load"))          opt.loadPath           = j["load"].get<std::string>();
    if (j.contains("log_csv"))       opt.logCsv             = j["log_csv"].get<bool>();
    if (j.contains("hidden"))        opt.hiddenOverride     = j["hidden"].get<int>();
    if (j.contains("episode_timeout")) opt.episodeTimeoutArg = j["episode_timeout"].get<float>();
    // Maps
    if (j.contains("train_maps")) {
        opt.trainMapsArg = jsonArrayToComma(j["train_maps"]);
        if (!opt.trainMapsArg.empty()) opt.explicitTrainMaps = true;
    }
    if (j.contains("val_maps"))      opt.valMapsArg         = jsonArrayToComma(j["val_maps"]);
    if (j.contains("test_maps"))     opt.testMapsArg        = jsonArrayToComma(j["test_maps"]);
    // Fitness aggregation
    if (j.contains("fitness_agg"))   opt.fitnessAggArg      = j["fitness_agg"].get<std::string>();
    if (j.contains("cvar_alpha"))    opt.cvarAlpha          = j["cvar_alpha"].get<float>();
    if (j.contains("map_norm"))      opt.mapNormArg         = j["map_norm"].get<std::string>();
    if (j.contains("map_weights"))   opt.mapWeightsArg      = jsonArrayToComma(j["map_weights"]);
    if (j.contains("progressive_frac")) opt.progressiveFrac = j["progressive_frac"].get<float>();
    if (j.contains("finetune_map"))  opt.finetuneMapArg     = j["finetune_map"].get<std::string>();
    // Curriculum
    if (j.contains("curriculum"))    opt.curriculumArg      = j["curriculum"].get<std::string>();
    if (j.contains("curriculum_start")) opt.currStart       = j["curriculum_start"].get<int>();
    if (j.contains("curriculum_step"))  opt.currStep        = j["curriculum_step"].get<int>();
    if (j.contains("curriculum_schedule")) opt.currScheduleArg = jsonArrayToComma(j["curriculum_schedule"]);
    if (j.contains("curriculum_pin")) opt.curriculumPinArg  = jsonArrayToComma(j["curriculum_pin"]);
    // Anti-overfitting
    if (j.contains("augment"))       opt.augmentArg         = jsonArrayToComma(j["augment"]);
    if (j.contains("dump_gen_maps")) opt.dumpGenMapsDir     = j["dump_gen_maps"].get<std::string>();
    if (j.contains("procedural_train")) opt.proceduralTrain = j["procedural_train"].get<int>();
    if (j.contains("procedural_val"))   opt.proceduralVal   = j["procedural_val"].get<int>();
    if (j.contains("proc_width_min"))   opt.procWidthMin    = j["proc_width_min"].get<float>();
    if (j.contains("proc_width_max"))   opt.procWidthMax    = j["proc_width_max"].get<float>();
    if (j.contains("random_spawn"))  opt.randomSpawn        = j["random_spawn"].get<bool>();
    if (j.contains("sensor_noise"))  opt.sensorNoise        = j["sensor_noise"].get<float>();
    if (j.contains("episodes_per_eval")) opt.episodesPerEval = j["episodes_per_eval"].get<int>();
    if (j.contains("episode_agg"))   opt.episodeAggArg      = j["episode_agg"].get<std::string>();
    // Model selection
    if (j.contains("select_by_val")) opt.selectByVal        = j["select_by_val"].get<bool>();
    if (j.contains("val_select_topk")) opt.valSelectTopK    = j["val_select_topk"].get<int>();
}

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "\nBasic\n"
        << "  --headless            Run without window\n"
        << "  --map <path>          Map JSON for watch/single-map train (default: maps/map1_chicanes_infernais.json)\n"
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
        << "                        Use --population N to race against N AIs (default: 1)\n"
        << "  --versus-noise <s>    Gaussian weight noise per AI copy so they diverge (default: 0.02)\n"
        << "\nMulti-map training & robustness\n"
        << "  --train-maps <a,b,..> Comma-separated list of training maps\n"
        << "  --val-maps <x,y>      Comma-separated validation maps (model selection when --select-by-val)\n"
        << "  --test-maps <x,y>     Comma-separated test maps: report-only, never used in selection\n"
        << "  --select-by-val       Save best.rnnw by validation progress (top-K) instead of train fitness\n"
        << "  --val-select-topk <T> Train genomes evaluated on val for selection (default 1)\n"
        << "  --random-spawn        Start each training episode at a random point on the track\n"
        << "  --sensor-noise <s>    Gaussian noise (stddev) added to ray readings in training\n"
        << "  --episodes-per-eval <N> Episodes per (genome,map), aggregated (default 1)\n"
        << "  --episode-agg <mode>  Combine episodes: mean (default) | min\n"
        << "  --augment <list>      Add transformed train maps: mirror,reverse,width:0.85,width:1.15\n"
        << "  --procedural-train <K> Generate K random tracks (seeded) into the train set\n"
        << "  --procedural-val <K>  Generate K random tracks (seeded) into the validation set\n"
        << "  --proc-width-min <w>  Min sampled width for procedural tracks (default 55)\n"
        << "  --proc-width-max <w>  Max sampled width for procedural tracks (default 110)\n"
        << "  --dump-gen-maps <dir> Save augmented+procedural train maps as JSON to <dir> before training\n"
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
        << "\nPerformance & map prioritization\n"
        << "  --episode-timeout SEC  Max episode length in seconds (default: 30)\n"
        << "  --curriculum-pin IDX  Comma-separated map indices always active (e.g. \"5\" or \"3,5\")\n"
        << "  --map-weights W1,...   Per-map weights for cvar-rank aggregation (e.g. \"1,1,1,1,1,3\")\n"
        << "                         Length must match --train-maps count; each weight > 0\n"
        << "  --progressive-frac F  Fraction of pop evaluated on maps[1..] after map[0] (default: 1.0)\n"
        << "  --finetune-map NAME   Fine-tune champion on a single map (requires --load)\n"
        << "                         Accepts bare name (map6_chicanes_infernais) or full path\n"
        << "\n  Default when --train is active and no --train-maps is given:\n"
        << "    all *.json in maps/, sorted; first 6 = train, last 2 = val (reproducible split).\n"
        << "  Single-map compat: --map without --train-maps trains only on that map.\n"
        << "\n  Baseline (old behaviour) reproducible with:\n"
        << "    --fitness-agg min --map-norm progress --curriculum none\n"
        << "\nConfig file\n"
        << "  --config <file.json>  Load defaults from JSON file (CLI flags override)\n"
        << "  Auto-detected: train.json in the current directory if it exists\n"
        << "  Keys match flag names with underscores: population, train_maps, cvar_alpha...\n"
        << "  Arrays accepted for train_maps, val_maps, test_maps, augment, map_weights\n";
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
    CliOptions opt;

    // ---- Config file: auto-detect train.json or use --config <file> -----------
    {
        std::string configPath;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--config" && i + 1 < argc) {
                configPath = argv[i + 1];
                break;
            }
        }
        if (configPath.empty() && std::filesystem::exists("train.json"))
            configPath = "train.json";

        if (!configPath.empty()) {
            std::ifstream f(configPath);
            if (!f) {
                std::cerr << "Cannot open config file: " << configPath << "\n";
                return 1;
            }
            try {
                json j = json::parse(f);
                applyConfig(j, opt);
                std::cerr << "[config] Loaded " << configPath << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error parsing config file " << configPath << ": " << e.what() << "\n";
                return 1;
            }
        }
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            if      (arg == "--headless")                    opt.sim.headless = true;
            else if (arg == "--benchmark")                   { opt.benchmark = true; opt.sim.headless = true; }
            else if (arg == "--train")                       opt.train           = true;
            else if (arg == "--map"          && i+1 < argc)  { opt.sim.map = argv[++i]; opt.explicitMap = true; }
            else if (arg == "--population"   && i+1 < argc)  opt.sim.population  = std::stoi(argv[++i]);
            else if (arg == "--seed"         && i+1 < argc)  opt.sim.seed        = (unsigned)std::stoul(argv[++i]);
            else if (arg == "--threads"      && i+1 < argc)  opt.sim.threads     = std::stoi(argv[++i]);
            else if (arg == "--algo"         && i+1 < argc)  opt.algo            = argv[++i];
            else if (arg == "--generations"  && i+1 < argc)  opt.generations     = std::stoi(argv[++i]);
            else if (arg == "--out"          && i+1 < argc)  opt.outDir          = argv[++i];
            else if (arg == "--load"         && i+1 < argc)  opt.loadPath        = argv[++i];
            else if (arg == "--watch"        && i+1 < argc)  opt.watchPath       = argv[++i];
            else if (arg == "--versus"       && i+1 < argc)  opt.versusPath      = argv[++i];
            else if (arg == "--versus-noise" && i+1 < argc)  opt.versusNoise     = std::stof(argv[++i]);
            else if (arg == "--train-maps"   && i+1 < argc)  { opt.trainMapsArg = argv[++i]; opt.explicitTrainMaps = true; }
            else if (arg == "--val-maps"     && i+1 < argc)  opt.valMapsArg      = argv[++i];
            else if (arg == "--test-maps"    && i+1 < argc)  opt.testMapsArg     = argv[++i];
            else if (arg == "--select-by-val")               opt.selectByVal     = true;
            else if (arg == "--val-select-topk" && i+1 < argc) opt.valSelectTopK = std::stoi(argv[++i]);
            else if (arg == "--random-spawn")                opt.randomSpawn     = true;
            else if (arg == "--sensor-noise"     && i+1 < argc) opt.sensorNoise   = std::stof(argv[++i]);
            else if (arg == "--episodes-per-eval" && i+1 < argc) opt.episodesPerEval = std::stoi(argv[++i]);
            else if (arg == "--episode-agg"      && i+1 < argc) opt.episodeAggArg  = argv[++i];
            else if (arg == "--augment"          && i+1 < argc) opt.augmentArg      = argv[++i];
            else if (arg == "--dump-gen-maps"    && i+1 < argc) opt.dumpGenMapsDir  = argv[++i];
            else if (arg == "--procedural-train" && i+1 < argc) opt.proceduralTrain = std::stoi(argv[++i]);
            else if (arg == "--procedural-val"   && i+1 < argc) opt.proceduralVal   = std::stoi(argv[++i]);
            else if (arg == "--proc-width-min"   && i+1 < argc) opt.procWidthMin    = std::stof(argv[++i]);
            else if (arg == "--proc-width-max"   && i+1 < argc) opt.procWidthMax    = std::stof(argv[++i]);
            else if (arg == "--fitness-agg"  && i+1 < argc)  opt.fitnessAggArg   = argv[++i];
            else if (arg == "--cvar-alpha"   && i+1 < argc)  opt.cvarAlpha       = std::stof(argv[++i]);
            else if (arg == "--map-norm"     && i+1 < argc)  opt.mapNormArg      = argv[++i];
            else if (arg == "--curriculum"   && i+1 < argc)  opt.curriculumArg   = argv[++i];
            else if (arg == "--curriculum-start" && i+1 < argc)  opt.currStart   = std::stoi(argv[++i]);
            else if (arg == "--curriculum-step"  && i+1 < argc)  opt.currStep    = std::stoi(argv[++i]);
            else if (arg == "--curriculum-schedule" && i+1 < argc) opt.currScheduleArg = argv[++i];
            else if (arg == "--hidden"       && i+1 < argc)  opt.hiddenOverride  = std::stoi(argv[++i]);
            else if (arg == "--log-csv")                      opt.logCsv         = true;
            else if (arg == "--episode-timeout"  && i+1 < argc) opt.episodeTimeoutArg = std::stof(argv[++i]);
            else if (arg == "--curriculum-pin"   && i+1 < argc) opt.curriculumPinArg  = argv[++i];
            else if (arg == "--map-weights"      && i+1 < argc) opt.mapWeightsArg     = argv[++i];
            else if (arg == "--progressive-frac" && i+1 < argc) opt.progressiveFrac   = std::stof(argv[++i]);
            else if (arg == "--finetune-map"     && i+1 < argc) opt.finetuneMapArg    = argv[++i];
            else if (arg == "--config"       && i+1 < argc) ++i; // already consumed above
            else if (arg == "--help" || arg == "-h")             opt.showHelp          = true;
            else { std::cerr << "Unknown option: " << arg << "\n"; opt.showHelp = true; }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing " << arg << ": " << e.what() << "\n";
            return 1;
        }
    }

    if (opt.showHelp) { printUsage(argv[0]); return 0; }

    // ---- Validate and parse multi-map config --------------------------------
    FitnessAgg   fitnessAgg = FitnessAgg::CVaRRank;
MapNormMode  mapNorm    = MapNormMode::ZScore;
    CurriculumMode currMode = CurriculumMode::Linear;

    try {
        fitnessAgg = parseAgg(opt.fitnessAggArg);
        mapNorm    = parseNorm(opt.mapNormArg);
        currMode   = parseCurriculum(opt.curriculumArg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (opt.cvarAlpha <= 0.f || opt.cvarAlpha > 1.f) {
        std::cerr << "--cvar-alpha must be in (0, 1], got " << opt.cvarAlpha << "\n";
        return 1;
    }

    // Warn if map-norm non-default but will be ignored under cvar-rank
    if (fitnessAgg == FitnessAgg::CVaRRank && mapNorm != MapNormMode::ZScore)
        std::cerr << "[warn] --map-norm is ignored under cvar-rank aggregation\n";

    // ---- Resolve --hidden and --load ----------------------------------------
    // Must be done before any NeuralNetwork construction.
    if (!opt.loadPath.empty()) {
        // Read topology from saved file to detect hidden size
        std::vector<int> topo;
        try {
            topo = readTopologyFromFile(opt.loadPath);
        } catch (const std::exception& e) {
            std::cerr << "Cannot read " << opt.loadPath << ": " << e.what() << "\n";
            return 1;
        }
        int fileH = (topo.size() >= 3) ? topo[1] : -1;

        if (opt.hiddenOverride >= 0) {
            if (opt.hiddenOverride != fileH) {
                std::cerr << "--hidden " << opt.hiddenOverride
                          << " conflicts with champion hidden size " << fileH
                          << " in " << opt.loadPath << "\n";
                return 1;
            }
            setHiddenSize(opt.hiddenOverride);
        } else {
            // Auto-detect from file topology
            if (fileH <= 0) {
                std::cerr << "Cannot infer hidden size from " << opt.loadPath
                          << " (topology has " << topo.size() << " layers)\n";
                return 1;
            }
            setHiddenSize(fileH);
        }
    } else if (opt.hiddenOverride >= 0) {
        setHiddenSize(opt.hiddenOverride);
    }

    // ---- Apply --episode-timeout ----
    if (opt.episodeTimeoutArg > 0.f) {
        if (opt.episodeTimeoutArg > 600.f) {
            std::cerr << "--episode-timeout must be in (0, 600], got " << opt.episodeTimeoutArg << "\n";
            return 1;
        }
        setEpisodeTimeout(opt.episodeTimeoutArg);
    }

    // ---- benchmark ----
    if (opt.benchmark) {
        Game::runBenchmark(opt.sim);
        return 0;
    }

    // ---- watch / load-without-train ----
    auto runWatch = [&](const std::string& path) {
#ifndef HEADLESS_ONLY
        std::vector<float> w = loadChampion(path);
        SimConfig wcfg = opt.sim;
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
        SimConfig vcfg = opt.sim;
        int nAI = std::max(1, opt.sim.population);
        vcfg.population = 1 + nAI;
        Game game(vcfg);
        {
            std::mt19937 rng(vcfg.seed);
            std::normal_distribution<float> noise(0.f, opt.versusNoise);
            std::vector<std::unique_ptr<AIController>> ctrls;
            ctrls.push_back(std::make_unique<HumanController>());
            for (int i = 0; i < nAI; ++i) {
                NeuralNetwork nn(defaultTopology());
                std::vector<float> wn = w;
                if (opt.versusNoise > 0.f)
                    for (float& x : wn) x += noise(rng);
                nn.setWeights(wn);
                ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
            }
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

    if (!opt.watchPath.empty()) { runWatch(opt.watchPath); return 0; }
    if (!opt.versusPath.empty()) { runVersus(opt.versusPath); return 0; }
    if (!opt.loadPath.empty() && !opt.train) { runWatch(opt.loadPath); return 0; }

    // ---- training ----
    if (opt.train) {
        // ----- Resolve train/val map lists -----------------------------------
        std::vector<std::string> trainMaps;
        std::vector<std::string> valMaps;

        if (opt.explicitTrainMaps) {
            trainMaps = splitComma(opt.trainMapsArg);
            if (!opt.valMapsArg.empty())
                valMaps = splitComma(opt.valMapsArg);
        } else if (opt.explicitMap) {
            trainMaps = { opt.sim.map };
            if (!opt.valMapsArg.empty())
                valMaps = splitComma(opt.valMapsArg);
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
            if (!opt.valMapsArg.empty())
                valMaps = splitComma(opt.valMapsArg);
        }

        if (trainMaps.empty()) {
            std::cerr << "No training maps found. Check maps/ directory or --train-maps.\n";
            return 1;
        }

        // Held-out test set: report-only, never used in selection.
        std::vector<std::string> testMaps;
        if (!opt.testMapsArg.empty())
            testMaps = splitComma(opt.testMapsArg);

        // ----- Validate curriculum-schedule ----------------------------------
        CurriculumConfig currCfg;
        currCfg.mode  = currMode;
        currCfg.start = opt.currStart;
        currCfg.step  = opt.currStep;

        if (currMode == CurriculumMode::Explicit) {
            if (opt.currScheduleArg.empty()) {
                std::cerr << "--curriculum explicit requires --curriculum-schedule <g1,g2,...>\n";
                return 1;
            }
            auto tokens = splitComma(opt.currScheduleArg);
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

        // ----- --curriculum-pin: parse and validate -------------------------
        if (!opt.curriculumPinArg.empty()) {
            auto tokens = splitComma(opt.curriculumPinArg);
            for (const auto& tok : tokens) {
                int idx = std::stoi(tok);
                if (idx < 0 || idx >= (int)trainMaps.size()) {
                    std::cerr << "--curriculum-pin index " << idx
                              << " out of range [0, " << trainMaps.size() - 1 << "]\n";
                    return 1;
                }
                currCfg.pinned.push_back(idx);
            }
        }

        // ----- --progressive-frac validation --------------------------------
        if (opt.progressiveFrac <= 0.f || opt.progressiveFrac > 1.f) {
            std::cerr << "--progressive-frac must be in (0, 1], got " << opt.progressiveFrac << "\n";
            return 1;
        }

        // ----- Build MultiMapConfig ------------------------------------------
        MultiMapConfig mmCfg;
        mmCfg.fitnessAgg      = fitnessAgg;
        mmCfg.cvarAlpha       = opt.cvarAlpha;
        mmCfg.mapNorm         = mapNorm;
        mmCfg.curriculum      = currCfg;
        mmCfg.progressiveFrac = opt.progressiveFrac;
        mmCfg.selectByVal     = opt.selectByVal;
        mmCfg.valSelectTopK   = opt.valSelectTopK;
        mmCfg.episodesPerEval = opt.episodesPerEval;
        mmCfg.randomSpawn     = opt.randomSpawn;
        mmCfg.sensorNoise     = opt.sensorNoise;
        mmCfg.episodeAgg      = (opt.episodeAggArg == "min") ? EpisodeAgg::Min : EpisodeAgg::Mean;

        if (opt.episodeAggArg != "mean" && opt.episodeAggArg != "min") {
            std::cerr << "--episode-agg must be 'mean' or 'min', got " << opt.episodeAggArg << "\n";
            return 1;
        }
        if (opt.episodesPerEval < 1) {
            std::cerr << "--episodes-per-eval must be >= 1, got " << opt.episodesPerEval << "\n";
            return 1;
        }
        if (opt.sensorNoise < 0.f) {
            std::cerr << "--sensor-noise must be >= 0, got " << opt.sensorNoise << "\n";
            return 1;
        }

        // --dump-gen-maps: create dir and pass to mmCfg
        if (!opt.dumpGenMapsDir.empty()) {
            std::filesystem::create_directories(opt.dumpGenMapsDir);
            mmCfg.dumpGenMaps = opt.dumpGenMapsDir;
        }

        // --procedural-train/val + optional width overrides
        mmCfg.proceduralTrain = opt.proceduralTrain;
        mmCfg.proceduralVal   = opt.proceduralVal;
        if (opt.proceduralTrain < 0 || opt.proceduralVal < 0) {
            std::cerr << "--procedural-train/--procedural-val must be >= 0\n";
            return 1;
        }
        if (opt.procWidthMin > 0.f) mmCfg.genParams.widthMin = opt.procWidthMin;
        if (opt.procWidthMax > 0.f) mmCfg.genParams.widthMax = opt.procWidthMax;
        if (mmCfg.genParams.widthMin > mmCfg.genParams.widthMax) {
            std::cerr << "--proc-width-min must be <= --proc-width-max\n";
            return 1;
        }

        // --augment: validate tokens (mirror | reverse | width:<factor>)
        if (!opt.augmentArg.empty()) {
            for (const auto& tok : splitComma(opt.augmentArg)) {
                if (tok == "mirror" || tok == "reverse") {
                    mmCfg.augment.push_back(tok);
                } else if (tok.rfind("width:", 0) == 0) {
                    float f = std::stof(tok.substr(6));
                    if (f <= 0.f) {
                        std::cerr << "--augment width factor must be > 0, got " << f << "\n";
                        return 1;
                    }
                    mmCfg.augment.push_back(tok);
                } else {
                    std::cerr << "--augment: unknown token '" << tok
                              << "' (expected mirror | reverse | width:<factor>)\n";
                    return 1;
                }
            }
        }

        if (opt.valSelectTopK < 1) {
            std::cerr << "--val-select-topk must be >= 1, got " << opt.valSelectTopK << "\n";
            return 1;
        }
        if (opt.selectByVal && valMaps.empty()) {
            std::cerr << "--select-by-val requires --val-maps (or an auto val split)\n";
            return 1;
        }

        // ----- --map-weights: parse and validate ----------------------------
        if (!opt.mapWeightsArg.empty()) {
            auto tokens = splitComma(opt.mapWeightsArg);
            if ((int)tokens.size() != (int)trainMaps.size()) {
                std::cerr << "--map-weights must have exactly " << trainMaps.size()
                          << " values (one per train map); got " << tokens.size() << "\n";
                return 1;
            }
            for (const auto& tok : tokens) {
                float w = std::stof(tok);
                if (w <= 0.f) {
                    std::cerr << "--map-weights: each weight must be > 0, got " << w << "\n";
                    return 1;
                }
                mmCfg.mapWeights.push_back(w);
            }
        }

        // ----- --finetune-map: override maps and force single-map mode ------
        if (!opt.finetuneMapArg.empty()) {
            if (opt.loadPath.empty()) {
                std::cerr << "--finetune-map requires --load <file.rnnw>\n";
                return 1;
            }
            // Accept bare name (e.g. "map6_chicanes_infernais") or full path
            std::string mapPath = opt.finetuneMapArg;
            if (mapPath.find('/') == std::string::npos && mapPath.find('\\') == std::string::npos)
                mapPath = "maps/" + mapPath + (mapPath.size() >= 5 && mapPath.substr(mapPath.size()-5) == ".json" ? "" : ".json");
            trainMaps = { mapPath };
            valMaps   = {};
            mmCfg.fitnessAgg          = FitnessAgg::Mean;
            mmCfg.curriculum.mode     = CurriculumMode::None;
            mmCfg.curriculum.pinned   = {};
            mmCfg.mapWeights          = {};
            mmCfg.progressiveFrac     = 1.0f;
            std::cout << "Fine-tune mode: training only on " << mapPath
                      << " from champion " << opt.loadPath << "\n";
        }

        // ----- Print config summary ------------------------------------------
        std::string aggName = opt.fitnessAggArg;
        if (fitnessAgg == FitnessAgg::CVaRRank || fitnessAgg == FitnessAgg::CVaRRaw)
            aggName += " α=" + std::to_string(opt.cvarAlpha);
        std::cout << "Training: " << trainMaps.size() << " train map(s)";
        if (!valMaps.empty()) std::cout << ", " << valMaps.size() << " val map(s)";
        std::cout << " | agg=" << aggName
                  << " | curriculum=" << opt.curriculumArg
                  << " | pop=" << opt.sim.population
                  << " | gen=" << opt.generations << "\n";
        for (size_t i = 0; i < trainMaps.size(); ++i)
            std::cout << "  train[" << i << "]: " << trainMaps[i] << "\n";
        for (size_t i = 0; i < valMaps.size(); ++i)
            std::cout << "  val["   << i << "]: " << valMaps[i]   << "\n";

        // ----- Champion seed -------------------------------------------------
        const std::vector<float>* champion = nullptr;
        std::vector<float> championWeights;
        if (!opt.loadPath.empty()) {
            championWeights = loadChampion(opt.loadPath);
            champion = &championWeights;
        }

        if (opt.sim.headless) {
            TrainingSession session(opt.sim, makeTrainer(opt.algo), opt.generations, opt.outDir,
                                    trainMaps, valMaps, mmCfg, champion, opt.logCsv, testMaps);
            session.runAll();
            return 0;
        }

#ifndef HEADLESS_ONLY
        bool multiMap = (trainMaps.size() > 1);
        TrainingSession session(opt.sim, makeTrainer(opt.algo), opt.generations, opt.outDir,
                                trainMaps, valMaps, mmCfg, champion, opt.logCsv, testMaps);
        Renderer renderer(900, 700, "assets/DejaVuSans.ttf");

        {
        auto mapDir = std::filesystem::path(opt.sim.map).parent_path().string();
        auto maps = listMaps(mapDir);
        int idx = findMapIndex(maps, opt.sim.map);

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
    if (opt.sim.headless) {
        Game game(opt.sim);
        double secs = game.runHeadlessEpisode();
        std::cout << "Headless episode done in " << secs << "s\n";
        return 0;
    }

    // ---- windowed default ----
#ifndef HEADLESS_ONLY
    Game game(opt.sim);
    {
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<HumanController>());
        std::mt19937 rng(opt.sim.seed + 1);
        for (int i = 1; i < opt.sim.population; ++i)
            ctrls.push_back(std::make_unique<NeuralNetworkController>(
                NeuralNetwork(defaultTopology(), (unsigned)rng())));
        game.setControllers(std::move(ctrls));
    }

    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
    game.reset();

    {
    auto mapDir = std::filesystem::path(opt.sim.map).parent_path().string();
    auto maps = listMaps(mapDir);
    int idx = findMapIndex(maps, opt.sim.map);

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