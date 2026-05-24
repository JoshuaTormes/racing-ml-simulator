#include "Track.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>
#include <filesystem>

using json = nlohmann::json;

Track::Track(const std::string& jsonPath) {
    std::ifstream f(jsonPath);
    if (!f) throw std::runtime_error("Track: cannot open " + jsonPath);

    json j;
    try { j = json::parse(f); }
    catch (const std::exception& e) {
        throw std::runtime_error("Track: JSON parse error in " + jsonPath + ": " + e.what());
    }

    if (!j.contains("waypoints") || !j.contains("track_width") || !j.contains("closed"))
        throw std::runtime_error("Track: missing required fields (waypoints/track_width/closed)");

    name_       = j.contains("name") ? j["name"].get<std::string>()
                                      : std::filesystem::path(jsonPath).stem().string();
    closed_     = j["closed"].get<bool>();
    trackWidth_ = j["track_width"].get<float>();

    for (auto& wp : j["waypoints"]) {
        if (!wp.is_array() || wp.size() < 2)
            throw std::runtime_error("Track: invalid waypoint entry");
        waypoints_.push_back({wp[0].get<float>(), wp[1].get<float>()});
    }
    if (waypoints_.size() < 3)
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
            obstacles_.push_back(o);
        }
    }

    buildBorders();
}

void Track::buildBorders() {
    int n = (int)waypoints_.size();
    leftBorder_.reserve(n + (closed_ ? 1 : 0));
    rightBorder_.reserve(n + (closed_ ? 1 : 0));
    float hw = trackWidth_ * 0.5f;

    for (int i = 0; i < n; ++i) {
        // Direction at this waypoint: average of incoming and outgoing segment directions
        Vec2 prev = waypoints_[(i - 1 + n) % n];
        Vec2 next = waypoints_[(i + 1) % n];
        Vec2 dir;
        if (!closed_ && i == 0)     dir = (waypoints_[1] - waypoints_[0]).normalized();
        else if (!closed_ && i == n-1) dir = (waypoints_[n-1] - waypoints_[n-2]).normalized();
        else dir = (next - prev).normalized();

        Vec2 perp = dir.perpendicular(); // left (upward) normal in y-down coords
        leftBorder_.push_back( waypoints_[i] + perp * hw);
        rightBorder_.push_back(waypoints_[i] - perp * hw);
    }

    if (closed_) {
        leftBorder_.push_back(leftBorder_[0]);
        rightBorder_.push_back(rightBorder_[0]);
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

float Track::raycast(Vec2 origin, Vec2 dir, float maxLen) const {
    float best = maxLen;

    auto checkPoly = [&](const std::vector<Vec2>& pts) {
        int n = (int)pts.size();
        for (int i = 0; i + 1 < n; ++i) {
            float t = segmentRaycast(origin, dir, pts[i], pts[i+1]);
            if (t < best) best = t;
        }
    };
    checkPoly(leftBorder_);
    checkPoly(rightBorder_);

    for (auto& ob : obstacles_) {
        if (ob.type == Obstacle::Type::Circle) {
            // Ray-circle
            Vec2  oc = origin - ob.pos;
            float b  = 2.f * oc.dot(dir);
            float c  = oc.dot(oc) - ob.radius * ob.radius;
            float disc = b * b - 4.f * c;
            if (disc >= 0.f) {
                float sq = std::sqrt(disc);
                float t0 = (-b - sq) * 0.5f;
                float t1 = (-b + sq) * 0.5f;
                float t  = (t0 >= 0.f) ? t0 : t1;
                if (t >= 0.f && t < best) best = t;
            }
        } else {
            // Ray-AABB (axis-aligned rect)
            float hx = ob.size.x * 0.5f, hy = ob.size.y * 0.5f;
            Vec2 corners[4] = {
                {ob.pos.x - hx, ob.pos.y - hy},
                {ob.pos.x + hx, ob.pos.y - hy},
                {ob.pos.x + hx, ob.pos.y + hy},
                {ob.pos.x - hx, ob.pos.y + hy}
            };
            for (int i = 0; i < 4; ++i) {
                float t = segmentRaycast(origin, dir, corners[i], corners[(i+1)%4]);
                if (t < best) best = t;
            }
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

bool Track::isInsideTrack(Vec2 p) const {
    int n = (int)waypoints_.size();
    int segs = (int)leftBorder_.size() - 1;
    // Check if point is within track_width/2 of any center segment
    float hw = trackWidth_ * 0.5f;
    for (int i = 0; i < segs; ++i) {
        // Use the midline between left and right border segments
        Vec2 cA = (leftBorder_[i]   + rightBorder_[i])   * 0.5f;
        Vec2 cB = (leftBorder_[i+1] + rightBorder_[i+1]) * 0.5f;
        if (pointNearSegment(p, cA, cB, hw)) return true;
    }
    // For open track, also handle last wp
    if (!closed_ && n >= 2) {
        Vec2 cA = (leftBorder_[n-2] + rightBorder_[n-2]) * 0.5f;
        Vec2 cB = (leftBorder_[n-1] + rightBorder_[n-1]) * 0.5f;
        if (pointNearSegment(p, cA, cB, hw)) return true;
    }
    return false;
}

Vec2 Track::spawnPos() const {
    return waypoints_[0];
}

float Track::spawnAngle() const {
    if (waypoints_.size() < 2) return 0.f;
    Vec2 d = waypoints_[1] - waypoints_[0];
    return std::atan2(d.y, d.x);
}

float Track::progressAt(Vec2 p, ProgressState& state) const {
    int n = (int)waypoints_.size();
    float prevProg = state.totalProg;
    float hw = trackWidth_ * 0.75f; // lookahead threshold

    // Advance nextWp while the car is close enough to the current target
    int limit = closed_ ? n : n - 1;
    while (true) {
        int idx = state.nextWp % n;
        float dist = (p - waypoints_[idx]).length();
        if (dist < hw && state.nextWp < limit + (closed_ ? 0 : 0)) {
            state.nextWp++;
            if (closed_) state.nextWp = ((state.nextWp - 1) % n) + 1;
        } else break;
    }

    // Fraction toward next waypoint
    int curIdx  = (state.nextWp - 1 + n) % n;
    int nextIdx = state.nextWp % n;
    Vec2 seg  = waypoints_[nextIdx] - waypoints_[curIdx];
    float len = seg.length();
    if (len < 1e-9f) {
        state.frac = 0.f;
    } else {
        float proj = (p - waypoints_[curIdx]).dot(seg.normalized());
        state.frac = std::clamp(proj / len, 0.f, 1.f);
    }

    int completedWps = state.nextWp - 1;
    state.totalProg = (float)completedWps + state.frac;

    return state.totalProg - prevProg;
}
