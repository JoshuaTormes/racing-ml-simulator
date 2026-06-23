#include "cli/ArgParser.h"
#include "cli/Util.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

// Apply values from a config JSON object to all CLI options.
// Called before the CLI parsing loop — subsequent CLI flags overwrite these values.
void applyConfig(const json& j, CliOptions& opt) {
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

} // namespace

ParseOutcome parseArgs(int argc, char** argv) {
    ParseOutcome out;
    CliOptions&  opt = out.opt;

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
                out.exit = true;
                out.code = 1;
                return out;
            }
            try {
                json j = json::parse(f);
                applyConfig(j, opt);
                std::cerr << "[config] Loaded " << configPath << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error parsing config file " << configPath << ": " << e.what() << "\n";
                out.exit = true;
                out.code = 1;
                return out;
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
            out.exit = true;
            out.code = 1;
            return out;
        }
    }

    return out;
}
