#pragma once
#include "core/Constants.h"
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

// Fitness = w_progress * maxProgress             (maxProgress ∈ [0, 1] — fraction of lap arc covered)
//           + speedBonus + checkpointBonus
//           - reversePenalty  (accumulated: w_reverse * (-speed) * DT while speed < 0)
//           - regressPenalty  (accumulated: w_regress * (maxProgress - currentProgress) * DT while behind peak)
//           - curvePenalty
//           (+ w_finish + w_time*(EPISODE_TIMEOUT - episodeTime) on completion)
//           (- w_crash on collision — each applied once)
// idle_eps: legacy field, kept for config compatibility (stall is now progress-based). Not used.
struct RewardConfig {
    // w_progress: weight on maxProgress (∈ [0,1] after the centerline refactor).
    // Old scale was per-waypoint (~10 wps → effective ~20 with w=2); new scale needs ~100×
    // multiplier so a full lap contributes comparable fitness.
    float w_progress = 200.0f;
    float w_finish   = 100.0f;
    float w_time     = 2.0f;
    float w_crash    = 15.0f;  // raised from 3 — crashing must clearly outweigh fast progress
    float idle_eps   = 5.0f;   // legacy — no longer used for stall detection
    // w_reverse: penalty per unit of reverse speed per second. 0 = disabled.
    // Only fires when speed < 0 (not when braking from positive speed). Range: [0, inf). Default: 0.5.
    float w_reverse  = 0.5f;
    // w_regress: penalty per unit of progress regression per second (maxProgress - currentProgress).
    // Makes reversing "expensive" in fitness terms. 0 = disabled. Range: [0, inf). Default: 1.0.
    float w_regress  = 2.0f;
    // w_speed: bonus per unit of (speed/MAX_SPEED) per second while making forward progress.
    // Incentivizes maintaining speed through straights without rewarding crashing fast. Range: [0, inf). Default: 0.3.
    float w_speed    = 0.3f;
    // w_checkpoint: one-shot bonus per design waypoint crossed, scaled by local centerline curvature.
    // Rewards navigating corners rather than stopping before them. Range: [0, inf). Default: 5.0.
    float w_checkpoint = 5.0f;
    // w_curve: penalty per unit of |curvature| * speed per second.
    // Directly pressures the GA to brake before tight corners. Range: [0, inf). Default: 0 (disabled).
    float w_curve    = 0.0f;
};

struct SimConfig {
    int         population = 1;
    unsigned    seed       = 42;
    bool        headless   = false;
    std::string map        = "maps/map1_chicanes_infernais.json";
    RewardConfig reward;
    int         threads    = 0; // 0 = hardware_concurrency
};
