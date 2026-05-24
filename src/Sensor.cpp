#include "Sensor.h"
#include "Track.h"
#include <cmath>

void Sensor::update(Vec2 pos, float angle, const Track& track) {
    // NUM_RAYS rays spread over 180° centered in front of the car
    float startAngle = angle - (float)M_PI * 0.5f;
    float step = (NUM_RAYS > 1) ? (float)M_PI / (NUM_RAYS - 1) : 0.f;

    for (int i = 0; i < NUM_RAYS; ++i) {
        float a = startAngle + step * i;
        Vec2 dir = {std::cos(a), std::sin(a)};
        float dist = track.raycast(pos, dir, RAY_MAX_LEN);
        readings_[i]  = dist / RAY_MAX_LEN; // normalized [0,1]
        hitPoints_[i] = pos + dir * dist;
    }
}
