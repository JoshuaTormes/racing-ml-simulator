#pragma once
#include "core/Vec2.h"
#include <vector>
#include <string>
#include <cstdint>

struct Obstacle {
    enum class Type { Circle, Rect } type;
    Vec2  pos;
    float radius = 0.f;    // Circle
    Vec2  size   = {0,0};  // Rect (half-extents stored internally as full size)
};

// Monotonic progress state per car (tracks which waypoint segment we're on)
struct ProgressState {
    int   nextWp    = 1;
    float frac      = 0.f;
    float totalProg = 0.f; // accumulated waypoint count + fraction
};

class Track {
public:
    explicit Track(const std::string& jsonPath);

    // Geometry queries (SFML-free)
    bool  isInsideTrack(Vec2 p) const;
    float raycast(Vec2 origin, Vec2 dir, float maxLen) const;
    // Returns delta progress this tick and updates state
    float progressAt(Vec2 p, ProgressState& state) const;

    // Spawn info
    Vec2  spawnPos()   const;
    float spawnAngle() const;

    // Accessors for Renderer / Sensor
    const std::vector<Vec2>&     waypoints()  const { return waypoints_; }
    const std::vector<Vec2>&     leftBorder() const { return leftBorder_; }
    const std::vector<Vec2>&     rightBorder()const { return rightBorder_; }
    const std::vector<Obstacle>& obstacles()  const { return obstacles_; }
    bool                         closed()     const { return closed_; }
    float                        trackWidth() const { return trackWidth_; }

private:
    std::vector<Vec2>     waypoints_;
    std::vector<Vec2>     leftBorder_;
    std::vector<Vec2>     rightBorder_;
    std::vector<Obstacle> obstacles_;
    bool  closed_      = true;
    float trackWidth_  = 120.f;

    void buildBorders();
    // Segment intersection helpers
    static float segmentRaycast(Vec2 ro, Vec2 rd, Vec2 a, Vec2 b);
    static bool  pointNearSegment(Vec2 p, Vec2 a, Vec2 b, float halfWidth);
};
