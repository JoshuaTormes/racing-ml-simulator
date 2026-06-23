#include "cli/AppConfig.h"
#include "cli/Util.h"
#include "control/NeuralNetwork.h"
#include "core/Constants.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

FitnessAgg parseAgg(const std::string& s) {
    if (s == "cvar-rank") return FitnessAgg::CVaRRank;
    if (s == "min")       return FitnessAgg::Min;
    if (s == "mean")      return FitnessAgg::Mean;
    if (s == "cvar-raw")  return FitnessAgg::CVaRRaw;
    throw std::runtime_error("Invalid --fitness-agg value: " + s +
                             ". Valid: cvar-rank | min | mean | cvar-raw");
}

MapNormMode parseNorm(const std::string& s) {
    if (s == "zscore")   return MapNormMode::ZScore;
    if (s == "minmax")   return MapNormMode::MinMax;
    if (s == "progress") return MapNormMode::Progress;
    throw std::runtime_error("Invalid --map-norm value: " + s +
                             ". Valid: zscore | minmax | progress");
}

CurriculumMode parseCurriculum(const std::string& s) {
    if (s == "none")     return CurriculumMode::None;
    if (s == "linear")   return CurriculumMode::Linear;
    if (s == "explicit") return CurriculumMode::Explicit;
    throw std::runtime_error("Invalid --curriculum value: " + s +
                             ". Valid: none | linear | explicit");
}

} // namespace

AppConfig resolveConfig(const CliOptions& opt) {
    AppConfig cfg;
    cfg.sim           = opt.sim;
    cfg.generations    = opt.generations;
    cfg.outDir         = opt.outDir;
    cfg.algo           = opt.algo;
    cfg.logCsv         = opt.logCsv;
    cfg.versusNoise    = opt.versusNoise;
    cfg.fitnessAggArg  = opt.fitnessAggArg;
    cfg.curriculumArg  = opt.curriculumArg;

    // ---- Validate and parse multi-map config --------------------------------
    FitnessAgg     fitnessAgg = parseAgg(opt.fitnessAggArg);
    MapNormMode    mapNorm    = parseNorm(opt.mapNormArg);
    CurriculumMode currMode   = parseCurriculum(opt.curriculumArg);

    if (opt.cvarAlpha <= 0.f || opt.cvarAlpha > 1.f) {
        std::ostringstream ss;
        ss << "--cvar-alpha must be in (0, 1], got " << opt.cvarAlpha;
        throw std::runtime_error(ss.str());
    }

    // Warn if map-norm non-default but will be ignored under cvar-rank
    if (fitnessAgg == FitnessAgg::CVaRRank && mapNorm != MapNormMode::ZScore)
        std::cerr << "[warn] --map-norm is ignored under cvar-rank aggregation\n";

    // ---- Resolve --hidden and --load ----------------------------------------
    // Must be done before any NeuralNetwork construction.
    if (!opt.loadPath.empty()) {
        std::vector<int> topo;
        try {
            topo = readTopologyFromFile(opt.loadPath);
        } catch (const std::exception& e) {
            std::ostringstream ss;
            ss << "Cannot read " << opt.loadPath << ": " << e.what();
            throw std::runtime_error(ss.str());
        }
        int fileH = (topo.size() >= 3) ? topo[1] : -1;

        if (opt.hiddenOverride >= 0) {
            if (opt.hiddenOverride != fileH) {
                std::ostringstream ss;
                ss << "--hidden " << opt.hiddenOverride
                   << " conflicts with champion hidden size " << fileH
                   << " in " << opt.loadPath;
                throw std::runtime_error(ss.str());
            }
            setHiddenSize(opt.hiddenOverride);
        } else {
            // Auto-detect from file topology
            if (fileH <= 0) {
                std::ostringstream ss;
                ss << "Cannot infer hidden size from " << opt.loadPath
                   << " (topology has " << topo.size() << " layers)";
                throw std::runtime_error(ss.str());
            }
            setHiddenSize(fileH);
        }
    } else if (opt.hiddenOverride >= 0) {
        setHiddenSize(opt.hiddenOverride);
    }

    // ---- Apply --episode-timeout ----
    if (opt.episodeTimeoutArg > 0.f) {
        if (opt.episodeTimeoutArg > 600.f) {
            std::ostringstream ss;
            ss << "--episode-timeout must be in (0, 600], got " << opt.episodeTimeoutArg;
            throw std::runtime_error(ss.str());
        }
        setEpisodeTimeout(opt.episodeTimeoutArg);
    }

    // ---- benchmark ----
    if (opt.benchmark) {
        cfg.mode = RunMode::Benchmark;
        return cfg;
    }

    // ---- watch / versus / load-without-train ----
    if (!opt.watchPath.empty()) {
        cfg.mode            = RunMode::Watch;
        cfg.watchSourcePath = opt.watchPath;
        return cfg;
    }
    if (!opt.versusPath.empty()) {
        cfg.mode       = RunMode::Versus;
        cfg.versusPath = opt.versusPath;
        return cfg;
    }
    if (!opt.loadPath.empty() && !opt.train) {
        cfg.mode            = RunMode::Watch;
        cfg.watchSourcePath = opt.loadPath;
        return cfg;
    }

    // ---- training ----
    if (opt.train) {
        cfg.mode = RunMode::Train;

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

        if (trainMaps.empty())
            throw std::runtime_error("No training maps found. Check maps/ directory or --train-maps.");

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
            if (opt.currScheduleArg.empty())
                throw std::runtime_error("--curriculum explicit requires --curriculum-schedule <g1,g2,...>");
            auto tokens = splitComma(opt.currScheduleArg);
            int expected = (int)trainMaps.size() - 1;
            if ((int)tokens.size() != expected) {
                std::ostringstream ss;
                ss << "--curriculum-schedule must have exactly M-1=" << expected
                   << " values for " << trainMaps.size() << " train maps; got "
                   << tokens.size();
                throw std::runtime_error(ss.str());
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
                    std::ostringstream ss;
                    ss << "--curriculum-pin index " << idx
                       << " out of range [0, " << trainMaps.size() - 1 << "]";
                    throw std::runtime_error(ss.str());
                }
                currCfg.pinned.push_back(idx);
            }
        }

        // ----- --progressive-frac validation --------------------------------
        if (opt.progressiveFrac <= 0.f || opt.progressiveFrac > 1.f) {
            std::ostringstream ss;
            ss << "--progressive-frac must be in (0, 1], got " << opt.progressiveFrac;
            throw std::runtime_error(ss.str());
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
            std::ostringstream ss;
            ss << "--episode-agg must be 'mean' or 'min', got " << opt.episodeAggArg;
            throw std::runtime_error(ss.str());
        }
        if (opt.episodesPerEval < 1) {
            std::ostringstream ss;
            ss << "--episodes-per-eval must be >= 1, got " << opt.episodesPerEval;
            throw std::runtime_error(ss.str());
        }
        if (opt.sensorNoise < 0.f) {
            std::ostringstream ss;
            ss << "--sensor-noise must be >= 0, got " << opt.sensorNoise;
            throw std::runtime_error(ss.str());
        }

        // --dump-gen-maps: create dir and pass to mmCfg
        if (!opt.dumpGenMapsDir.empty()) {
            std::filesystem::create_directories(opt.dumpGenMapsDir);
            mmCfg.dumpGenMaps = opt.dumpGenMapsDir;
        }

        // --procedural-train/val + optional width overrides
        mmCfg.proceduralTrain = opt.proceduralTrain;
        mmCfg.proceduralVal   = opt.proceduralVal;
        if (opt.proceduralTrain < 0 || opt.proceduralVal < 0)
            throw std::runtime_error("--procedural-train/--procedural-val must be >= 0");
        if (opt.procWidthMin > 0.f) mmCfg.genParams.widthMin = opt.procWidthMin;
        if (opt.procWidthMax > 0.f) mmCfg.genParams.widthMax = opt.procWidthMax;
        if (mmCfg.genParams.widthMin > mmCfg.genParams.widthMax)
            throw std::runtime_error("--proc-width-min must be <= --proc-width-max");

        // --augment: validate tokens (mirror | reverse | width:<factor>)
        if (!opt.augmentArg.empty()) {
            for (const auto& tok : splitComma(opt.augmentArg)) {
                if (tok == "mirror" || tok == "reverse") {
                    mmCfg.augment.push_back(tok);
                } else if (tok.rfind("width:", 0) == 0) {
                    float f = std::stof(tok.substr(6));
                    if (f <= 0.f) {
                        std::ostringstream ss;
                        ss << "--augment width factor must be > 0, got " << f;
                        throw std::runtime_error(ss.str());
                    }
                    mmCfg.augment.push_back(tok);
                } else {
                    std::ostringstream ss;
                    ss << "--augment: unknown token '" << tok
                       << "' (expected mirror | reverse | width:<factor>)";
                    throw std::runtime_error(ss.str());
                }
            }
        }

        if (opt.valSelectTopK < 1) {
            std::ostringstream ss;
            ss << "--val-select-topk must be >= 1, got " << opt.valSelectTopK;
            throw std::runtime_error(ss.str());
        }
        if (opt.selectByVal && valMaps.empty())
            throw std::runtime_error("--select-by-val requires --val-maps (or an auto val split)");

        // ----- --map-weights: parse and validate ----------------------------
        if (!opt.mapWeightsArg.empty()) {
            auto tokens = splitComma(opt.mapWeightsArg);
            if ((int)tokens.size() != (int)trainMaps.size()) {
                std::ostringstream ss;
                ss << "--map-weights must have exactly " << trainMaps.size()
                   << " values (one per train map); got " << tokens.size();
                throw std::runtime_error(ss.str());
            }
            for (const auto& tok : tokens) {
                float w = std::stof(tok);
                if (w <= 0.f) {
                    std::ostringstream ss;
                    ss << "--map-weights: each weight must be > 0, got " << w;
                    throw std::runtime_error(ss.str());
                }
                mmCfg.mapWeights.push_back(w);
            }
        }

        // ----- --finetune-map: override maps and force single-map mode ------
        if (!opt.finetuneMapArg.empty()) {
            if (opt.loadPath.empty())
                throw std::runtime_error("--finetune-map requires --load <file.rnnw>");
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

        // ----- Champion seed -------------------------------------------------
        if (!opt.loadPath.empty()) {
            cfg.championWeights = loadChampion(opt.loadPath);
            cfg.hasChampion     = true;
        }

        cfg.trainMaps = trainMaps;
        cfg.valMaps   = valMaps;
        cfg.testMaps  = testMaps;
        cfg.mmCfg     = mmCfg;
        return cfg;
    }

    // ---- default (headless / windowed) ----
    cfg.mode = cfg.sim.headless ? RunMode::HeadlessDefault : RunMode::WindowedDefault;
    return cfg;
}
