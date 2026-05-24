#pragma once

constexpr int   SIM_HZ          = 60;
constexpr float DT              = 1.f / SIM_HZ;
constexpr int   NUM_RAYS        = 7;
constexpr float RAY_MAX_LEN     = 300.f;
constexpr float MAX_SPEED       = 400.f;
constexpr float ACCEL           = 300.f;
constexpr float BRAKE           = 500.f;
constexpr float DRAG            = 0.98f;
constexpr float MAX_STEER       = 3.0f;
constexpr float EPISODE_TIMEOUT = 60.f;
constexpr float STALL_TIMEOUT   = 5.f;
constexpr int   OBS_SIZE        = NUM_RAYS + 3; // 10
