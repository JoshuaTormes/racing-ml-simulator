#pragma once
#include "core/Vec2.h"
#include "core/Types.h"
#include "Track.h"
#include "Sensor.h"

enum class DoneReason { None, Collision, Timeout, Stall, Completed };

class Car {
public:
    // Public state (read by Game/Renderer)
    Vec2  pos         = {0, 0};
    float angle       = 0.f;
    float speed       = 0.f;
    float fitness     = 0.f;
    float idleTime    = 0.f;
    float episodeTime = 0.f;
    bool  done        = false;
    DoneReason doneReason = DoneReason::None;
    ProgressState progState;
    Sensor sensor;

    void reset(Vec2 spawnPos, float spawnAngle);

    void applyAction(const Action& a);

    Observation observe(const Track& track) const;

    // Advance physics, evaluate done conditions; returns reward for this tick
    float stepDone(const Track& track, const RewardConfig& cfg);

private:
    float prevProgress_ = 0.f;
};
