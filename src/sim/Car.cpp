#include "sim/Car.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

void Car::reset(Vec2 spawnPos, float spawnAngle) {
    reset(spawnPos, spawnAngle, 0.f);
}

void Car::reset(Vec2 spawnPos, float spawnAngle, float spawnArcFrac_) {
    pos             = spawnPos;
    angle           = spawnAngle;
    spawnArcFrac    = spawnArcFrac_;
    speed           = 0.f;
    fitness         = 0.f;
    maxProgress     = 0.f;
    idleTime        = 0.f;
    episodeTime     = 0.f;
    reversePenalty  = 0.f;
    regressPenalty  = 0.f;
    curvePenalty    = 0.f;
    speedBonus      = 0.f;
    checkpointBonus = 0.f;
    lastCheckpoint  = 1;  // skip design waypoint 0 (spawn)
    noProgressTime      = 0.f;
    progressAtLastReset = 0.f;
    lowSpeedTime        = 0.f;
    done            = false;
    doneReason      = DoneReason::None;
    projState       = ProjectionState{};
}

void Car::applyAction(const Action& a) {
    // Throttle: positive = accelerate, negative = brake
    float accelForce = (a.throttle > 0.f) ?  a.throttle * ACCEL * DT
                                           :  a.throttle * BRAKE * DT;
    speed += accelForce;
    speed *= DRAG;
    speed = std::clamp(speed, MAX_REVERSE_SPEED, MAX_SPEED);

    float yawCmd  = std::fabs(a.steering) * MAX_STEER;
    float yawGrip = MAX_LAT_ACCEL / std::max(std::fabs(speed), 1.f);
    float yawRate = std::copysign(std::min(yawCmd, yawGrip), a.steering);
    angle += yawRate * DT;

    pos.x += std::cos(angle) * speed * DT;
    pos.y += std::sin(angle) * speed * DT;
}

Observation Car::observe(const Track& track) const {
    Observation obs{};
    // Rays (sensor was updated in last stepDone)
    for (int i = 0; i < NUM_RAYS; ++i)
        obs[i] = sensor.readings()[i];

    // Normalized speed
    obs[NUM_RAYS] = std::fabs(speed) / MAX_SPEED;

    // Centerline projection at current position
    Vec2  projPoint = track.centerlineAtArc(projState.arcLen);
    Vec2  tangent   = track.tangentAtArc(projState.arcLen);
    Vec2  normal    = tangent.perpendicular(); // leftward in y-down screen space

    // Lateral offset: positive = right of centerline. Vec2::perpendicular() is leftward,
    // so we negate to make positive = right (matches spec convention).
    Vec2  diff = pos - projPoint;
    float lateral = -diff.dot(normal) / (track.trackWidth() * 0.5f);
    obs[NUM_RAYS + 1] = std::clamp(lateral, -1.f, 1.f);

    // Heading error: car angle vs track tangent, wrapped to [-π, π], normalized by π.
    float trackHeading = std::atan2(tangent.y, tangent.x);
    float herr = angle - trackHeading;
    while (herr >  (float)M_PI) herr -= 2.f * (float)M_PI;
    while (herr < -(float)M_PI) herr += 2.f * (float)M_PI;
    obs[NUM_RAYS + 2] = herr / (float)M_PI;

    // Lookaheads: 5 × (signed curvature, speed_excess) at fixed arc distances ahead.
    for (int i = 0; i < NUM_LOOKAHEADS; ++i) {
        float arcAhead = projState.arcLen + LOOKAHEAD_ARCS[i];
        float ang = track.curvatureAtArc(arcAhead);              // signed radians
        float radius = (std::fabs(ang) > 1e-3f)
            ? CURVATURE_ARC_WINDOW / std::fabs(ang)
            : 9999.f;
        float safe_v = std::min(std::sqrt(MAX_LAT_ACCEL * radius), MAX_SPEED);
        float excess = std::clamp((std::fabs(speed) - safe_v) / MAX_SPEED, -1.f, 1.f);

        obs[NUM_RAYS + 3 + 2 * i]     = ang / (float)M_PI;
        obs[NUM_RAYS + 3 + 2 * i + 1] = excess;
    }

    return obs;
}

float Car::stepDone(const Track& track, const RewardConfig& cfg) {
    if (done) return 0.f;

    float fitnessBefore = fitness;

    episodeTime += DT;

    // Update sensors
    sensor.update(pos, angle, track);

    // Update projection onto the dense centerline.
    // Cumulative progress (lap + arcLen/totalArc) is robust to spurious wrap-shortcuts
    // near spawn — a bad projection that jumps the lap counter doesn't inflate progress.
    track.updateProjection(pos, projState);
    float totalArc = track.totalArcLength();
    float curProgress = (totalArc > 0.f)
        ? (float)projState.lap + projState.arcLen / totalArc
        : 0.f;
    // Measure progress relative to the spawn point. With spawnArcFrac == 0 (default /
    // fixed-spawn path) this is a no-op; with a random mid-track spawn it ensures the
    // car earns no free progress at tick 0 and must drive a full lap from where it began.
    curProgress -= spawnArcFrac;
    if (curProgress > maxProgress)
        maxProgress = curProgress;

    // Stall timer resets only on meaningful forward advance.
    if (maxProgress >= progressAtLastReset + STALL_PROGRESS_MIN) {
        progressAtLastReset = maxProgress;
        noProgressTime      = 0.f;
    } else {
        noProgressTime += DT;
    }

    // Bonus G: one-shot reward when crossing each design waypoint, scaled by local curvature.
    // Only fires while the car is on its first proper lap (lap == 0). Lap > 0 means a full
    // lap was completed and all waypoints already credited; lap < 0 is a spurious projection
    // wrap that we silently ignore.
    if (projState.lap == 0) {
        const auto& cpArcs = track.checkpointArcLens();
        int nCheckpoints = (int)cpArcs.size();
        while (lastCheckpoint < nCheckpoints && projState.arcLen >= cpArcs[lastCheckpoint]) {
            float curv = std::fabs(track.curvatureAtArc(cpArcs[lastCheckpoint]) / (float)M_PI);
            checkpointBonus += cfg.w_checkpoint * curv;
            ++lastCheckpoint;
        }
    }

    // Penalty B: accumulate for each tick spent moving in reverse
    if (speed < 0.f)
        reversePenalty += cfg.w_reverse * (-speed) * DT;

    // Penalty C: accumulate for each tick spent behind the progress peak
    float regress = maxProgress - curProgress;
    if (regress > 0.f)
        regressPenalty += cfg.w_regress * regress * DT;

    // Upcoming curvature (shared by bonus F and penalty E) — short arc lookahead
    float curvatureAhead = std::fabs(track.curvatureAtArc(projState.arcLen + 60.f) / (float)M_PI);

    // Bonus F: speed bonus only on straights
    if (curvatureAhead < 0.15f && curProgress >= maxProgress)
        speedBonus += cfg.w_speed * (std::fabs(speed) / MAX_SPEED) * DT;

    // Penalty E: high speed through tight corners (disabled by default, w_curve=0)
    curvePenalty += cfg.w_curve * curvatureAhead * std::fabs(speed) * DT;

    // Fitness: progress + bonuses minus accumulated penalties (overwritten each tick)
    fitness = cfg.w_progress * maxProgress + speedBonus + checkpointBonus
            - reversePenalty - regressPenalty - curvePenalty;

    // Check done conditions
    // Completed: high-water mark of normalized progress reached 99% of the lap.
    // In closed mode this fires before the wrap to 0; in open mode at the endpoint.
    if (maxProgress >= 0.99f) {
        done = true; doneReason = DoneReason::Completed;
        fitness += cfg.w_finish + cfg.w_time * (EPISODE_TIMEOUT - episodeTime);
        return fitness - fitnessBefore;
    }
    // Timeout
    if (episodeTime >= EPISODE_TIMEOUT) {
        done = true; doneReason = DoneReason::Timeout;
        return fitness - fitnessBefore;
    }
    // Speed stall (independent of progress-based stall — catches spin-in-place)
    if (std::fabs(speed) < STALL_SPEED) lowSpeedTime += DT;
    else                                lowSpeedTime  = 0.f;

    if (noProgressTime >= STALL_TIMEOUT || lowSpeedTime >= STALL_TIMEOUT) {
        done = true; doneReason = DoneReason::Stall;
        return fitness - fitnessBefore;
    }
    // Collision — check center + 4 corners of the car rectangle
    {
        float ca = std::cos(angle), sa = std::sin(angle);
        float hl = CAR_LENGTH * 0.5f, hw_c = CAR_WIDTH * 0.5f;
        Vec2 pts[5] = {
            pos,
            {pos.x + ca*hl - sa*hw_c, pos.y + sa*hl + ca*hw_c},
            {pos.x + ca*hl + sa*hw_c, pos.y + sa*hl - ca*hw_c},
            {pos.x - ca*hl - sa*hw_c, pos.y - sa*hl + ca*hw_c},
            {pos.x - ca*hl + sa*hw_c, pos.y - sa*hl - ca*hw_c},
        };
        bool outside = false;
        for (auto& p : pts) outside = outside || !track.isInsideTrack(p);
        if (outside) {
            done = true; doneReason = DoneReason::Collision;
            fitness -= cfg.w_crash;
            return fitness - fitnessBefore;
        }
    }

    return fitness - fitnessBefore;
}
