#pragma once
#include "Constants.h"
#include <array>
#include <string>

using Observation = std::array<float, OBS_SIZE>;

struct Action {
    float throttle = 0.f; // [-1, 1]: positive = accelerate, negative = brake/reverse
    float steering = 0.f; // [-1, 1]: negative = left, positive = right
};

struct StepResult {
    Observation next;
    float reward;
    bool done;
};

// Configurable reward weights (all fields overridable)
struct RewardConfig {
    float w_progress = 1.0f;
    float w_speed    = 0.1f;
    float w_idle     = 0.05f;
    float w_finish   = 100.0f;
    float w_crash    = 50.0f;
    float idle_eps   = 5.0f; // speed threshold for idle penalty (px/s)
};

struct SimConfig {
    int         population = 1;
    unsigned    seed       = 42;
    bool        headless   = false;
    std::string map        = "maps/map1.json";
    RewardConfig reward;
    int         threads    = 0; // 0 = hardware_concurrency
};
