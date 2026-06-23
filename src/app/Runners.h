#pragma once
#include "cli/AppConfig.h"

int runBenchmark(const AppConfig& cfg);
int runWatch(const AppConfig& cfg);
int runVersus(const AppConfig& cfg);
int runInteractive(const AppConfig& cfg);
int runHeadlessDefault(const AppConfig& cfg);
int runTraining(const AppConfig& cfg);

// Dispatches to the runner matching cfg.mode. Each runner that touches
// Renderer/SFML guards its real implementation behind #ifndef HEADLESS_ONLY
// internally (see Runners.cpp) — they remain callable from any build.
int dispatch(const AppConfig& cfg);
