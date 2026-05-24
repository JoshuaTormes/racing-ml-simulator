#pragma once
#include "core/Vec2.h"
#include "core/Constants.h"
#include <array>

class Track;

class Sensor {
public:
    // Cast rays from pos at angle, populate readings and hit points
    void update(Vec2 pos, float angle, const Track& track);

    const std::array<float, NUM_RAYS>& readings()   const { return readings_; }
    const std::array<Vec2,  NUM_RAYS>& hitPoints()  const { return hitPoints_; }

private:
    std::array<float, NUM_RAYS> readings_{};
    std::array<Vec2,  NUM_RAYS> hitPoints_{};
};
