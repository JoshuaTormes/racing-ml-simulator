#pragma once
#include "core/Types.h"
#include <string>

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