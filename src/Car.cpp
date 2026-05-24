#include "Car.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

void Car::reset(Vec2 spawnPos, float spawnAngle) {
    pos          = spawnPos;
    angle        = spawnAngle;
    speed        = 0.f;
    fitness      = 0.f;
    idleTime     = 0.f;
    episodeTime  = 0.f;
    done         = false;
    doneReason   = DoneReason::None;
    progState    = ProgressState{};
    prevProgress_= 0.f;
}

void Car::applyAction(const Action& a) {
    // Throttle: positive = accelerate, negative = brake
    float accelForce = (a.throttle > 0.f) ?  a.throttle * ACCEL * DT
                                           : -a.throttle * BRAKE * DT;
    // Negative throttle brakes/reverses
    speed += accelForce;
    speed *= DRAG;
    speed = std::clamp(speed, -MAX_SPEED, MAX_SPEED);

    // Steering scaled by speed (so steer is weaker at low speed)
    float steerRate = a.steering * MAX_STEER * (std::fabs(speed) / MAX_SPEED);
    angle += steerRate * DT;

    // Integrate position
    pos.x += std::cos(angle) * speed * DT;
    pos.y += std::sin(angle) * speed * DT;
}

Observation Car::observe(const Track& track) const {
    // Sensor was already updated in stepDone; use its readings
    Observation obs{};
    for (int i = 0; i < NUM_RAYS; ++i)
        obs[i] = sensor.readings()[i];

    // speed normalized to [0,1]
    obs[NUM_RAYS] = std::fabs(speed) / MAX_SPEED;

    // angle to next waypoint, normalized to [-1,1]
    int n = (int)track.waypoints().size();
    int nextIdx = progState.nextWp % n;
    Vec2 toWp = track.waypoints()[nextIdx] - pos;
    float wpAngle = std::atan2(toWp.y, toWp.x);
    float relAngle = wpAngle - angle;
    // Wrap to [-pi, pi]
    while (relAngle >  (float)M_PI) relAngle -= 2.f * (float)M_PI;
    while (relAngle < -(float)M_PI) relAngle += 2.f * (float)M_PI;
    obs[NUM_RAYS + 1] = relAngle / (float)M_PI; // [-1, 1]

    // distance to next waypoint normalized [0,1]
    float dist = toWp.length();
    obs[NUM_RAYS + 2] = std::min(dist / 400.f, 1.f); // normalize over ~400px

    return obs;
}

float Car::stepDone(const Track& track, const RewardConfig& cfg) {
    if (done) return 0.f;

    episodeTime += DT;

    // Update sensors
    sensor.update(pos, angle, track);

    // Update progress
    float deltaProgress = track.progressAt(pos, progState);

    // Track idle (no speed progress)
    if (std::fabs(speed) < cfg.idle_eps) idleTime += DT;
    else idleTime = 0.f;

    // Compute reward for this tick
    float reward = 0.f;
    reward += cfg.w_progress * deltaProgress;
    reward += cfg.w_speed    * (std::fabs(speed) / MAX_SPEED);
    if (std::fabs(speed) < cfg.idle_eps) reward -= cfg.w_idle;

    // Check done conditions
    int n = (int)track.waypoints().size();
    bool isCircuit = track.closed();
    int  totalWps  = isCircuit ? n : n - 1;

    // Completed circuit
    if (progState.nextWp > totalWps) {
        done = true; doneReason = DoneReason::Completed;
        reward += cfg.w_finish;
        fitness += reward;
        return reward;
    }
    // Timeout
    if (episodeTime >= EPISODE_TIMEOUT) {
        done = true; doneReason = DoneReason::Timeout;
        fitness += reward;
        return reward;
    }
    // Stall
    if (idleTime >= STALL_TIMEOUT) {
        done = true; doneReason = DoneReason::Stall;
        fitness += reward;
        return reward;
    }
    // Collision
    if (!track.isInsideTrack(pos)) {
        done = true; doneReason = DoneReason::Collision;
        reward -= cfg.w_crash;
        fitness += reward;
        return reward;
    }

    fitness += reward;
    return reward;
}
