#pragma once
#include "cli/CliOptions.h"
#include "training/Training.h"
#include <string>
#include <vector>

enum class RunMode { Benchmark, Watch, Versus, Train, HeadlessDefault, WindowedDefault };

// Fully resolved & validated configuration consumed by the mode runners
// (src/app/Runners.h). Built from CliOptions by resolveConfig().
struct AppConfig {
    RunMode   mode = RunMode::WindowedDefault;
    SimConfig sim;

    // Watch / versus
    std::string watchSourcePath; // .rnnw to load for RunMode::Watch
    std::string versusPath;      // .rnnw to load for RunMode::Versus
    float       versusNoise = 0.02f;

    // Training
    std::vector<std::string> trainMaps;
    std::vector<std::string> valMaps;
    std::vector<std::string> testMaps;
    MultiMapConfig            mmCfg;
    int                       generations = 100;
    std::string               outDir      = "out/";
    std::string               algo        = "genetic";
    bool                      logCsv      = false;

    std::vector<float> championWeights;
    bool                hasChampion = false;

    // Raw strings echoed verbatim in the training config summary printed by the runner.
    std::string fitnessAggArg;
    std::string curriculumArg;
};

// Resolves and validates a CliOptions into a domain-ready AppConfig. All validation
// failures that used to print to std::cerr and `return 1` now throw std::runtime_error
// with the identical message (caught once in main()).
AppConfig resolveConfig(const CliOptions& opt);
