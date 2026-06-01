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

// Plain description of a track, decoupled from any file format. Lets a Track be
// built in memory (procedural generation, augmentation) using the same geometry
// pipeline as the JSON path. See Track::loadData() and Track(const TrackData&).
struct TrackData {
    std::vector<Vec2>     waypoints;
    float                 trackWidth = 120.f;
    bool                  closed     = true;
    std::string           name;
    std::vector<Obstacle> obstacles;
};

// Per-car projection onto the dense centerline.
// segIdx + t identifies the location within one lap; arcLen is the cumulative arc
// length within the current lap. `lap` counts full laps (can go negative when the
// local search spuriously wraps near spawn — see updateProjection comments).
// Cumulative progress: lap * totalArcLength + arcLen.
struct ProjectionState {
    int   segIdx         = 0;
    float t              = 0.f;
    float arcLen         = 0.f;
    int   lap            = 0;
};

class Track {
public:
    explicit Track(const std::string& jsonPath);
    // Build a track in memory (no file). Runs the same centerline/border/index
    // pipeline as the JSON ctor. Throws on <3 waypoints or non-positive width.
    explicit Track(const TrackData& data);

    // Parse a map JSON into a TrackData without building geometry. Reused by the
    // JSON ctor and by augmentation/procedural code that transforms the data.
    static TrackData loadData(const std::string& jsonPath);
    // Serialize a TrackData back to JSON file (same format as loadData).
    static void saveData(const TrackData& d, const std::string& jsonPath);

    // Geometry queries (SFML-free)
    bool  isInsideTrack(Vec2 p) const;
    float raycast(Vec2 origin, Vec2 dir, float maxLen) const;

    // Projects p onto the dense centerline via local search; updates state.
    // Returns delta normalized progress this tick (∈ ~[-1, 1], wrap-corrected
    // in closed mode). Absolute progress is state.arcLen / totalArcLength_.
    float updateProjection(Vec2 p, ProjectionState& state) const;

    // Arc-parametrised queries (closed: arc is wrapped; open: clamped to [0, L])
    Vec2  centerlineAtArc(float arc) const;
    Vec2  tangentAtArc(float arc) const;     // unit vector
    float curvatureAtArc(float arc) const;   // signed radians; positive = right-turn (y-down screen)

    // Spawn info (derived from the dense centerline)
    Vec2  spawnPos()   const;
    float spawnAngle() const;

    // Accessors for Renderer / Sensor / consumers
    // waypoints(): kept for compat — now returns the DENSE centerline so that
    // border-aware geometry (raycast, isInsideTrack) and downstream loops keep working.
    // Use designWaypoints() to iterate over the original JSON anchors (sparse).
    const std::vector<Vec2>&     waypoints()         const { return centerline_; }
    const std::vector<Vec2>&     designWaypoints()   const { return designWaypoints_; }
    const std::vector<Vec2>&     centerline()        const { return centerline_; }
    const std::vector<Vec2>&     leftBorder()        const { return leftBorder_; }
    const std::vector<Vec2>&     rightBorder()       const { return rightBorder_; }
    const std::vector<Obstacle>& obstacles()         const { return obstacles_; }
    bool                         closed()            const { return closed_; }
    float                        trackWidth()        const { return trackWidth_; }
    const std::string&           name()              const { return name_; }
    const std::vector<float>&    arcLength()         const { return arcLength_; }
    float                        totalArcLength()    const { return totalArcLength_; }
    const std::vector<float>&    checkpointArcLens() const { return checkpointArcLens_; }

private:
    struct GridCell {
        std::vector<int> segIdxLeft;
        std::vector<int> segIdxRight;
        std::vector<int> quadIdx;
        std::vector<int> obstacleIdx;
    };

    std::vector<Vec2>     designWaypoints_;
    std::vector<Vec2>     centerline_;
    std::vector<float>    arcLength_;
    std::vector<float>    checkpointArcLens_;
    float                 totalArcLength_ = 0.f;
    std::vector<Vec2>     leftBorder_;
    std::vector<Vec2>     rightBorder_;
    std::vector<Obstacle> obstacles_;
    bool  closed_      = true;
    float trackWidth_  = 120.f;
    std::string name_;

    std::vector<GridCell> grid_;
    Vec2  gridOrigin_    = {0.f, 0.f};
    float gridCellSize_  = 0.f;
    int   gridW_         = 0;
    int   gridH_         = 0;

    void buildCenterline();
    void buildBorders();
    void buildSpatialIndex();

    int   segmentIndexForArc(float arc) const;
    float wrapArc(float arc) const;

    static float segmentRaycast(Vec2 ro, Vec2 rd, Vec2 a, Vec2 b);
    static bool  pointNearSegment(Vec2 p, Vec2 a, Vec2 b, float halfWidth);
};
