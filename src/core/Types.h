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

// Fitness = w_progress * maxProgress + w_speed * avgSpeed
//           (+ w_finish on completion, - w_crash on collision — each applied once)
// idle_eps: speed threshold for stall detection (px/s), not a fitness weight
struct RewardConfig {
    float w_progress = 1.0f;
    float w_speed    = 2.0f;
    float w_finish   = 100.0f;
    float w_crash    = 5.0f;
    float idle_eps   = 5.0f;
};

struct SimConfig {
    int         population = 1;
    unsigned    seed       = 42;
    bool        headless   = false;
    std::string map        = "maps/map1.json";
    RewardConfig reward;
    int         threads    = 0; // 0 = hardware_concurrency
};
