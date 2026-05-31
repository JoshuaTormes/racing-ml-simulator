#include "Track.h"
#include "core/Constants.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>
#include <filesystem>

using json = nlohmann::json;

TrackData Track::loadData(const std::string& jsonPath) {
    std::ifstream f(jsonPath);
    if (!f) throw std::runtime_error("Track: cannot open " + jsonPath);

    json j;
    try { j = json::parse(f); }
    catch (const std::exception& e) {
        throw std::runtime_error("Track: JSON parse error in " + jsonPath + ": " + e.what());
    }

    if (!j.contains("waypoints") || !j.contains("track_width") || !j.contains("closed"))
        throw std::runtime_error("Track: missing required fields (waypoints/track_width/closed)");

    TrackData d;
    d.name       = j.contains("name") ? j["name"].get<std::string>()
                                      : std::filesystem::path(jsonPath).stem().string();
    d.closed     = j["closed"].get<bool>();
    d.trackWidth = j["track_width"].get<float>();

    for (auto& wp : j["waypoints"]) {
        if (!wp.is_array() || wp.size() < 2)
            throw std::runtime_error("Track: invalid waypoint entry");
        d.waypoints.push_back({wp[0].get<float>(), wp[1].get<float>()});
    }
    if (d.waypoints.size() < 3)
        throw std::runtime_error("Track: need at least 3 waypoints");

    if (j.contains("obstacles")) {
        for (auto& ob : j["obstacles"]) {
            Obstacle o;
            std::string t = ob["type"].get<std::string>();
            auto pos = ob["pos"];
            o.pos = {pos[0].get<float>(), pos[1].get<float>()};
            if (t == "circle") {
                o.type   = Obstacle::Type::Circle;
                o.radius = ob["radius"].get<float>();
            } else if (t == "rect") {
                o.type = Obstacle::Type::Rect;
                auto sz = ob["size"];
                o.size = {sz[0].get<float>(), sz[1].get<float>()};
            } else {
                throw std::runtime_error("Track: unknown obstacle type: " + t);
            }
            d.obstacles.push_back(o);
        }
    }
    return d;
}

// JSON ctor delegates to the in-memory ctor after parsing.
Track::Track(const std::string& jsonPath) : Track(loadData(jsonPath)) {}

Track::Track(const TrackData& data) {
    if (data.waypoints.size() < 3)
        throw std::runtime_error("Track: need at least 3 waypoints");
    if (data.trackWidth <= 0.f)
        throw std::runtime_error("Track: track_width must be > 0");

    name_            = data.name;
    closed_          = data.closed;
    trackWidth_      = data.trackWidth;
    designWaypoints_ = data.waypoints;
    obstacles_       = data.obstacles;

    buildCenterline();
    buildBorders();
    buildSpatialIndex();
}

// ---------------------------------------------------------------------------
// Centripetal Catmull-Rom (alpha=0.5) interpolation between p1 (u=0) and p2 (u=1)
// using Barry-Goldman pyramidal form. p0 and p3 set the tangent at p1 and p2.
// ---------------------------------------------------------------------------
static Vec2 catmullRomCentripetal(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float u) {
    auto knot = [](float ti, Vec2 a, Vec2 b) -> float {
        float dx = b.x - a.x, dy = b.y - a.y;
        float d = std::sqrt(std::sqrt(dx*dx + dy*dy)); // alpha=0.5
        if (d < 1e-6f) d = 1e-6f; // protect against coincident points
        return ti + d;
    };
    float t0 = 0.f;
    float t1 = knot(t0, p0, p1);
    float t2 = knot(t1, p1, p2);
    float t3 = knot(t2, p2, p3);
    float t  = t1 + u * (t2 - t1);

    auto lerp = [](Vec2 a, Vec2 b, float w) -> Vec2 {
        return {a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w};
    };

    Vec2 A1 = lerp(p0, p1, (t - t0) / (t1 - t0));
    Vec2 A2 = lerp(p1, p2, (t - t1) / (t2 - t1));
    Vec2 A3 = lerp(p2, p3, (t - t2) / (t3 - t2));
    Vec2 B1 = lerp(A1, A2, (t - t0) / (t2 - t0));
    Vec2 B2 = lerp(A2, A3, (t - t1) / (t3 - t1));
    Vec2 C  = lerp(B1, B2, (t - t1) / (t2 - t1));
    return C;
}

void Track::buildCenterline() {
    int n = (int)designWaypoints_.size();
    const int N = CENTERLINE_SUBSEGMENTS;
    centerline_.clear();
    arcLength_.clear();
    checkpointArcLens_.clear();
    centerline_.reserve(closed_ ? (size_t)n * N + 1 : (size_t)(n - 1) * N + 1);

    // wp(i): index into designWaypoints_ with closed-mode wrap or open-mode reflection.
    // Open: reflect endpoints through neighbour so the tangent at the endpoint matches
    // the segment direction (avoids overshoot at the start/end).
    auto wp = [&](int i) -> Vec2 {
        if (closed_) return designWaypoints_[((i % n) + n) % n];
        if (i < 0)   return { 2.f * designWaypoints_[0].x - designWaypoints_[1].x,
                              2.f * designWaypoints_[0].y - designWaypoints_[1].y };
        if (i >= n)  return { 2.f * designWaypoints_[n-1].x - designWaypoints_[n-2].x,
                              2.f * designWaypoints_[n-1].y - designWaypoints_[n-2].y };
        return designWaypoints_[i];
    };

    int trechos = closed_ ? n : (n - 1);
    for (int i = 0; i < trechos; ++i) {
        Vec2 p0 = wp(i - 1);
        Vec2 p1 = wp(i);
        Vec2 p2 = wp(i + 1);
        Vec2 p3 = wp(i + 2);
        for (int k = 0; k < N; ++k) {
            float u = (float)k / (float)N;
            centerline_.push_back(catmullRomCentripetal(p0, p1, p2, p3, u));
        }
    }
    // Final endpoint: in closed mode duplicate wp[0] to close the loop;
    // in open mode include wp[n-1] (which was skipped by t<1 sampling).
    centerline_.push_back(closed_ ? designWaypoints_[0]
                                  : designWaypoints_[n - 1]);

    // Pre-compute cumulative arc length per centerline point.
    arcLength_.resize(centerline_.size(), 0.f);
    for (size_t i = 1; i < centerline_.size(); ++i) {
        float d = (centerline_[i] - centerline_[i - 1]).length();
        arcLength_[i] = arcLength_[i - 1] + d;
    }
    totalArcLength_ = arcLength_.back();

    // Map each designWaypoint -> arc length. Design wp k corresponds to
    // centerline_[k * N] (segment start). The closure point (closed mode) is
    // at centerline_[n*N] = designWaypoints_[0]; we don't index it again here.
    checkpointArcLens_.resize(n);
    for (int k = 0; k < n; ++k) {
        int idx = k * N;
        if (idx > (int)arcLength_.size() - 1) idx = (int)arcLength_.size() - 1;
        checkpointArcLens_[k] = arcLength_[idx];
    }
}

void Track::buildBorders() {
    int n = (int)centerline_.size();
    leftBorder_.clear();
    rightBorder_.clear();
    leftBorder_.reserve(n);
    rightBorder_.reserve(n);
    float hw = trackWidth_ * 0.5f;
    // Miter limit: prevents border spikes/crossings at sharp turns.
    // At a 90° turn the miter is ~1.41×hw; cap at 3× to handle hairpins gracefully.
    const float miterLimit = 3.0f;

    // In closed mode centerline_[n-1] == centerline_[0]; the wrap below uses
    // index (i-1+n-1) % (n-1) so we treat the loop as having (n-1) unique points.
    auto prevIdx = [&](int i) {
        if (i > 0) return i - 1;
        return closed_ ? (n - 2) : 0;
    };
    auto nextIdx = [&](int i) {
        if (i + 1 < n) return i + 1;
        return closed_ ? 1 : (n - 1);
    };

    for (int i = 0; i < n; ++i) {
        Vec2 d1, d2;
        if (!closed_ && i == 0) {
            d1 = d2 = (centerline_[1] - centerline_[0]).normalized();
        } else if (!closed_ && i == n - 1) {
            d1 = d2 = (centerline_[n - 1] - centerline_[n - 2]).normalized();
        } else {
            d1 = (centerline_[i] - centerline_[prevIdx(i)]).normalized();
            d2 = (centerline_[nextIdx(i)] - centerline_[i]).normalized();
        }

        Vec2 n1 = d1.perpendicular();
        Vec2 n2 = d2.perpendicular();
        Vec2 miter = n1 + n2;
        float miterLen2 = miter.lengthSq();

        Vec2 offset;
        if (miterLen2 < 1e-6f) {
            offset = n2 * hw;
        } else {
            miter = miter * (1.0f / std::sqrt(miterLen2));
            float dotVal = miter.dot(n1);
            if (dotVal < 1e-4f) {
                offset = n2 * hw;
            } else {
                float scale = std::min(hw / dotVal, hw * miterLimit);
                offset = miter * scale;
            }
        }

        leftBorder_.push_back( centerline_[i] + offset);
        rightBorder_.push_back(centerline_[i] - offset);
    }
}

// ---------------------------------------------------------------------------
// Spatial index: uniform grid over the track AABB.
// Cell size = RAY_MAX_LEN / 8 ≈ 50 units.
// ---------------------------------------------------------------------------
void Track::buildSpatialIndex() {
    if (leftBorder_.empty() || rightBorder_.empty()) return;

    const float margin = RAY_MAX_LEN * 0.1f;
    float minX = leftBorder_[0].x, maxX = minX;
    float minY = leftBorder_[0].y, maxY = minY;

    auto expandAABB = [&](Vec2 p) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    };
    for (auto& p : leftBorder_)  expandAABB(p);
    for (auto& p : rightBorder_) expandAABB(p);
    for (auto& ob : obstacles_) {
        float rx = (ob.type == Obstacle::Type::Circle) ? ob.radius : ob.size.x * 0.5f;
        float ry = (ob.type == Obstacle::Type::Circle) ? ob.radius : ob.size.y * 0.5f;
        expandAABB({ob.pos.x - rx, ob.pos.y - ry});
        expandAABB({ob.pos.x + rx, ob.pos.y + ry});
    }
    minX -= margin; minY -= margin;
    maxX += margin; maxY += margin;

    gridCellSize_ = RAY_MAX_LEN / 8.f;
    gridOrigin_   = {minX, minY};
    gridW_ = std::max(1, (int)std::ceil((maxX - minX) / gridCellSize_));
    gridH_ = std::max(1, (int)std::ceil((maxY - minY) / gridCellSize_));
    grid_.assign((size_t)(gridW_ * gridH_), GridCell{});

    // Helper: mark a range of cells for a given segment/quad/obstacle.
    auto markCells = [&](float x0f, float y0f, float x1f, float y1f,
                         std::vector<int> GridCell::* field, int idx) {
        int cx0 = std::max(0, (int)((x0f - gridOrigin_.x) / gridCellSize_));
        int cx1 = std::min(gridW_ - 1, (int)((x1f - gridOrigin_.x) / gridCellSize_));
        int cy0 = std::max(0, (int)((y0f - gridOrigin_.y) / gridCellSize_));
        int cy1 = std::min(gridH_ - 1, (int)((y1f - gridOrigin_.y) / gridCellSize_));
        for (int gy = cy0; gy <= cy1; ++gy)
            for (int gx = cx0; gx <= cx1; ++gx)
                (grid_[(size_t)(gy * gridW_ + gx)].*field).push_back(idx);
    };

    int segs = (int)leftBorder_.size() - 1;
    for (int i = 0; i < segs; ++i) {
        // Left border segment
        {
            Vec2 a = leftBorder_[i], b = leftBorder_[i+1];
            markCells(std::min(a.x, b.x), std::min(a.y, b.y),
                      std::max(a.x, b.x), std::max(a.y, b.y),
                      &GridCell::segIdxLeft, i);
        }
        // Right border segment
        {
            Vec2 a = rightBorder_[i], b = rightBorder_[i+1];
            markCells(std::min(a.x, b.x), std::min(a.y, b.y),
                      std::max(a.x, b.x), std::max(a.y, b.y),
                      &GridCell::segIdxRight, i);
        }
        // Full quad AABB (for isInsideTrack)
        {
            Vec2 pts[4] = {leftBorder_[i], leftBorder_[i+1],
                           rightBorder_[i], rightBorder_[i+1]};
            float qx0 = pts[0].x, qx1 = pts[0].x;
            float qy0 = pts[0].y, qy1 = pts[0].y;
            for (int k = 1; k < 4; ++k) {
                qx0 = std::min(qx0, pts[k].x); qx1 = std::max(qx1, pts[k].x);
                qy0 = std::min(qy0, pts[k].y); qy1 = std::max(qy1, pts[k].y);
            }
            markCells(qx0, qy0, qx1, qy1, &GridCell::quadIdx, i);
        }
    }

    for (int oi = 0; oi < (int)obstacles_.size(); ++oi) {
        const auto& ob = obstacles_[(size_t)oi];
        float rx = (ob.type == Obstacle::Type::Circle) ? ob.radius : ob.size.x * 0.5f;
        float ry = (ob.type == Obstacle::Type::Circle) ? ob.radius : ob.size.y * 0.5f;
        markCells(ob.pos.x - rx, ob.pos.y - ry,
                  ob.pos.x + rx, ob.pos.y + ry,
                  &GridCell::obstacleIdx, oi);
    }
}

// Ray vs segment, returns t in [0, maxLen] or infinity
float Track::segmentRaycast(Vec2 ro, Vec2 rd, Vec2 a, Vec2 b) {
    Vec2  ab  = b - a;
    float det = rd.x * ab.y - rd.y * ab.x;
    if (std::fabs(det) < 1e-9f) return std::numeric_limits<float>::infinity();

    Vec2  ao = ro - a;
    float t  = -(ao.x * ab.y - ao.y * ab.x) / det;
    float u  = -(ao.x * rd.y - ao.y * rd.x) / det;
    if (t < 0.f || u < 0.f || u > 1.f) return std::numeric_limits<float>::infinity();
    return t;
}

static void testObstacleRay(Vec2 origin, Vec2 dir, const Obstacle& ob, float& best) {
    if (ob.type == Obstacle::Type::Circle) {
        Vec2  oc   = origin - ob.pos;
        float b    = 2.f * oc.dot(dir);
        float c    = oc.dot(oc) - ob.radius * ob.radius;
        float disc = b * b - 4.f * c;
        if (disc >= 0.f) {
            float sq = std::sqrt(disc);
            float t0 = (-b - sq) * 0.5f;
            float t1 = (-b + sq) * 0.5f;
            float t  = (t0 >= 0.f) ? t0 : t1;
            if (t >= 0.f && t < best) best = t;
        }
    } else {
        float hx = ob.size.x * 0.5f, hy = ob.size.y * 0.5f;
        Vec2 corners[4] = {
            {ob.pos.x - hx, ob.pos.y - hy},
            {ob.pos.x + hx, ob.pos.y - hy},
            {ob.pos.x + hx, ob.pos.y + hy},
            {ob.pos.x - hx, ob.pos.y + hy}
        };
        for (int k = 0; k < 4; ++k) {
            Vec2 a = corners[k], bb = corners[(k+1)%4];
            Vec2 ab = bb - a;
            float det = dir.x * ab.y - dir.y * ab.x;
            if (std::fabs(det) < 1e-9f) continue;
            Vec2  ao = origin - a;
            float t  = -(ao.x * ab.y - ao.y * ab.x) / det;
            float u  = -(ao.x * dir.y - ao.y * dir.x) / det;
            if (t >= 0.f && u >= 0.f && u <= 1.f && t < best) best = t;
        }
    }
}

float Track::raycast(Vec2 origin, Vec2 dir, float maxLen) const {
    float best = maxLen;

    // DDA traversal through the spatial grid.
    float ox = origin.x - gridOrigin_.x;
    float oy = origin.y - gridOrigin_.y;
    int cx = (int)(ox / gridCellSize_);
    int cy = (int)(oy / gridCellSize_);

    // Brute-force fallback when origin is outside the grid.
    if (gridW_ == 0 || cx < 0 || cx >= gridW_ || cy < 0 || cy >= gridH_) {
        int n = (int)leftBorder_.size();
        for (int i = 0; i + 1 < n; ++i) {
            float t = segmentRaycast(origin, dir, leftBorder_[i], leftBorder_[i+1]);
            if (t < best) best = t;
            t = segmentRaycast(origin, dir, rightBorder_[i], rightBorder_[i+1]);
            if (t < best) best = t;
        }
        for (auto& ob : obstacles_) testObstacleRay(origin, dir, ob, best);
        return best;
    }

    int stepX = 0, stepY = 0;
    float tMaxX = std::numeric_limits<float>::infinity();
    float tMaxY = std::numeric_limits<float>::infinity();
    float tDeltaX = std::numeric_limits<float>::infinity();
    float tDeltaY = std::numeric_limits<float>::infinity();

    if (std::fabs(dir.x) >= 1e-9f) {
        stepX = (dir.x > 0.f) ? 1 : -1;
        float boundX = gridOrigin_.x + (cx + (stepX > 0 ? 1 : 0)) * gridCellSize_;
        tMaxX   = (boundX - origin.x) / dir.x;
        tDeltaX = gridCellSize_ / std::fabs(dir.x);
    }
    if (std::fabs(dir.y) >= 1e-9f) {
        stepY = (dir.y > 0.f) ? 1 : -1;
        float boundY = gridOrigin_.y + (cy + (stepY > 0 ? 1 : 0)) * gridCellSize_;
        tMaxY   = (boundY - origin.y) / dir.y;
        tDeltaY = gridCellSize_ / std::fabs(dir.y);
    }

    // Guard against fully-zero direction (degenerate ray).
    if (stepX == 0 && stepY == 0) return best;

    float tEntry = 0.f;
    while (cx >= 0 && cx < gridW_ && cy >= 0 && cy < gridH_ && tEntry < best) {
        const GridCell& cell = grid_[(size_t)(cy * gridW_ + cx)];

        for (int si : cell.segIdxLeft) {
            float t = segmentRaycast(origin, dir, leftBorder_[(size_t)si], leftBorder_[(size_t)si + 1]);
            if (t < best) best = t;
        }
        for (int si : cell.segIdxRight) {
            float t = segmentRaycast(origin, dir, rightBorder_[(size_t)si], rightBorder_[(size_t)si + 1]);
            if (t < best) best = t;
        }
        for (int oi : cell.obstacleIdx)
            testObstacleRay(origin, dir, obstacles_[(size_t)oi], best);

        if (tMaxX < tMaxY) {
            tEntry = tMaxX;
            cx    += stepX;
            tMaxX += tDeltaX;
        } else {
            tEntry = tMaxY;
            cy    += stepY;
            tMaxY += tDeltaY;
        }
    }

    return best;
}

bool Track::pointNearSegment(Vec2 p, Vec2 a, Vec2 b, float halfWidth) {
    Vec2  ab = b - a;
    float len2 = ab.lengthSq();
    if (len2 < 1e-9f) return (p - a).length() < halfWidth;
    float t = std::clamp((p - a).dot(ab) / len2, 0.f, 1.f);
    Vec2 closest = a + ab * t;
    return (p - closest).length() < halfWidth;
}

static bool inTriangle(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    auto cross2d = [](Vec2 u, Vec2 v) { return u.x * v.y - u.y * v.x; };
    float d0 = cross2d(b - a, p - a);
    float d1 = cross2d(c - b, p - b);
    float d2 = cross2d(a - c, p - c);
    bool allPos = (d0 >= 0) && (d1 >= 0) && (d2 >= 0);
    bool allNeg = (d0 <= 0) && (d1 <= 0) && (d2 <= 0);
    return allPos || allNeg;
}

bool Track::isInsideTrack(Vec2 p) const {
    float rx = p.x - gridOrigin_.x;
    float ry = p.y - gridOrigin_.y;
    int cx = (int)(rx / gridCellSize_);
    int cy = (int)(ry / gridCellSize_);

    if (gridW_ == 0 || cx < 0 || cx >= gridW_ || cy < 0 || cy >= gridH_) {
        // Brute-force fallback for out-of-grid points.
        int segs = (int)leftBorder_.size() - 1;
        for (int i = 0; i < segs; ++i) {
            Vec2 L0 = leftBorder_[i],  L1 = leftBorder_[i+1];
            Vec2 R0 = rightBorder_[i], R1 = rightBorder_[i+1];
            if (inTriangle(p, L0, L1, R0) || inTriangle(p, L1, R1, R0))
                return true;
        }
        return false;
    }

    const GridCell& cell = grid_[(size_t)(cy * gridW_ + cx)];
    for (int i : cell.quadIdx) {
        Vec2 L0 = leftBorder_[(size_t)i],    L1 = leftBorder_[(size_t)i + 1];
        Vec2 R0 = rightBorder_[(size_t)i],   R1 = rightBorder_[(size_t)i + 1];
        if (inTriangle(p, L0, L1, R0) || inTriangle(p, L1, R1, R0))
            return true;
    }
    return false;
}

Vec2 Track::spawnPos() const {
    return designWaypoints_[0];
}

float Track::spawnAngle() const {
    Vec2 t = tangentAtArc(0.f);
    return std::atan2(t.y, t.x);
}

// ---------------------------------------------------------------------------
// Arc-parametrised helpers
// ---------------------------------------------------------------------------
float Track::wrapArc(float arc) const {
    if (totalArcLength_ <= 0.f) return 0.f;
    if (closed_) {
        float L = totalArcLength_;
        float a = std::fmod(arc, L);
        if (a < 0.f) a += L;
        return a;
    }
    return std::clamp(arc, 0.f, totalArcLength_);
}

int Track::segmentIndexForArc(float arc) const {
    // Returns i ∈ [0, centerline_.size()-2] such that arcLength_[i] <= arc < arcLength_[i+1].
    int lastSeg = (int)arcLength_.size() - 2;
    if (lastSeg < 0) return 0;
    if (arc >= arcLength_[lastSeg + 1]) return lastSeg;
    int lo = 0, hi = lastSeg;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (arcLength_[mid] <= arc) lo = mid;
        else                        hi = mid - 1;
    }
    return lo;
}

Vec2 Track::centerlineAtArc(float arc) const {
    arc = wrapArc(arc);
    int i = segmentIndexForArc(arc);
    float segLen = arcLength_[i + 1] - arcLength_[i];
    float t = (segLen > 1e-9f) ? (arc - arcLength_[i]) / segLen : 0.f;
    return centerline_[i] + (centerline_[i + 1] - centerline_[i]) * t;
}

Vec2 Track::tangentAtArc(float arc) const {
    arc = wrapArc(arc);
    int i = segmentIndexForArc(arc);
    Vec2 d = centerline_[i + 1] - centerline_[i];
    return d.normalized();
}

float Track::curvatureAtArc(float arc) const {
    // Signed angle between tangent at `arc` and tangent at `arc + CURVATURE_ARC_WINDOW`.
    // ∈ approximately [-π, π]; positive = right-turn in y-down screen space.
    Vec2 t1 = tangentAtArc(arc);
    Vec2 t2 = tangentAtArc(arc + CURVATURE_ARC_WINDOW);
    float cr = t1.x * t2.y - t1.y * t2.x;
    float dt = t1.x * t2.x + t1.y * t2.y;
    return std::atan2(cr, dt);
}

// ---------------------------------------------------------------------------
// Local-search projection with explicit lap tracking.
// Search window: ±K segments around state.segIdx, indices outside [0, segCount)
// wrap (closed mode only) with a corresponding lap delta. Tracking lap separately
// from arcLen means that when the search spuriously snaps to the wrap-equivalent
// endpoint at spawn, cumulative progress (lap * L + arcLen) stays put — preventing
// false completion. State.segIdx is visited first; strict-less updates bias toward
// continuity so true on-centerline projection wins ties.
// ---------------------------------------------------------------------------
float Track::updateProjection(Vec2 p, ProjectionState& state) const {
    int segCount = (int)centerline_.size() - 1;
    if (segCount <= 0) return 0.f;

    const int K = 16;

    // rawIdx may be anywhere in [-K + state.segIdx, +K + state.segIdx]; wraps map
    // to (idx ∈ [0, segCount), lapDelta ∈ {-1, 0, +1}).
    auto evalRaw = [&](int rawIdx,
                       float& outD2, int& outSeg, float& outT, int& outLapDelta) -> bool {
        int idx = rawIdx;
        outLapDelta = 0;
        if (idx < 0) {
            if (!closed_) return false;
            outLapDelta = -1; idx += segCount;
        } else if (idx >= segCount) {
            if (!closed_) return false;
            outLapDelta = +1; idx -= segCount;
        }
        if (idx < 0 || idx >= segCount) return false;  // double-wrap not allowed

        Vec2 a = centerline_[idx];
        Vec2 b = centerline_[idx + 1];
        Vec2 ab = b - a;
        float len2 = ab.lengthSq();
        outT = (len2 > 1e-12f) ? std::clamp((p - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
        Vec2 closest = a + ab * outT;
        outD2 = (p - closest).lengthSq();
        outSeg = idx;
        return true;
    };

    float bestD2 = std::numeric_limits<float>::infinity();
    int   bestSeg = state.segIdx;
    float bestT = state.t;
    int   bestLapDelta = 0;

    {
        float d2, t; int seg, ld;
        if (evalRaw(state.segIdx, d2, seg, t, ld)) {
            bestD2 = d2; bestSeg = seg; bestT = t; bestLapDelta = ld;
        }
    }
    for (int dk = 1; dk <= K; ++dk) {
        for (int sign : {+1, -1}) {
            float d2, t; int seg, ld;
            if (evalRaw(state.segIdx + sign * dk, d2, seg, t, ld)) {
                if (d2 < bestD2) {
                    bestD2 = d2; bestSeg = seg; bestT = t; bestLapDelta = ld;
                }
            }
        }
    }

    float prevTotal = (float)state.lap * totalArcLength_ + state.arcLen;
    state.lap   += bestLapDelta;
    state.segIdx = bestSeg;
    state.t      = bestT;
    state.arcLen = arcLength_[bestSeg] + bestT * (arcLength_[bestSeg + 1] - arcLength_[bestSeg]);
    float newTotal = (float)state.lap * totalArcLength_ + state.arcLen;

    return totalArcLength_ > 0.f ? (newTotal - prevTotal) / totalArcLength_ : 0.f;
}
