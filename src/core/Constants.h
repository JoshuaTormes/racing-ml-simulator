#pragma once

constexpr int   SIM_HZ          = 60;
constexpr float DT              = 1.f / SIM_HZ;
constexpr int   NUM_RAYS        = 7;
constexpr float RAY_MAX_LEN     = 400.f;
constexpr float MAX_SPEED         = 400.f;
// Maximum reverse speed (px/s). 0 = reverse fully disabled (negative throttle acts as pure brake).
// Set to a small negative value (e.g. -40) to allow minor reversing to unstick the car. Range: [-MAX_SPEED, 0].
constexpr float MAX_REVERSE_SPEED = 0.f;
constexpr float ACCEL           = 300.f;
constexpr float BRAKE           = 500.f;
constexpr float DRAG            = 0.98f;
constexpr float MAX_STEER       = 3.0f;
constexpr float MAX_LAT_ACCEL   = 650.f;  // grip limit (px/s²): yawRate ≤ MAX_LAT_ACCEL/v
constexpr float EPISODE_TIMEOUT = 60.f;
constexpr float STALL_TIMEOUT      = 2.f;
constexpr float STALL_SPEED        = 4.f;    // px/s — below this counts as stopped
constexpr float STALL_PROGRESS_MIN = 0.05f;  // min waypoint advance to reset stall timer (blocks spin-in-place)
// inputs: rays[0..NUM_RAYS-1], speed, angle_to_wp, dist_to_wp, angle_to_wp2,
//         then 5x (curvature_Nahead, speed_excess_Nahead) for N=1..5
constexpr int   OBS_SIZE        = NUM_RAYS + 14; // 21
constexpr int   NN_HIDDEN       = 32;
constexpr int   ACT_SIZE        = 2;
constexpr float CAR_LENGTH      = 20.f;
constexpr float CAR_WIDTH       = 10.f;
