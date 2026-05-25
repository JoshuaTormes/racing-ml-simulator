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
    leftBorder_.clear();
    rightBorder_.clear();
    leftBorder_.reserve(n + (closed_ ? 1 : 0));
    rightBorder_.reserve(n + (closed_ ? 1 : 0));
    float hw = trackWidth_ * 0.5f;
    // Miter limit: prevents border spikes/crossings at sharp turns.
    // At a 90° turn the miter is ~1.41×hw; cap at 3× to handle hairpins gracefully.
    const float miterLimit = 3.0f;

    for (int i = 0; i < n; ++i) {
        Vec2 d1, d2;
        if (!closed_ && i == 0) {
            d1 = d2 = (waypoints_[1] - waypoints_[0]).normalized();
        } else if (!closed_ && i == n - 1) {
            d1 = d2 = (waypoints_[n-1] - waypoints_[n-2]).normalized();
        } else {
            d1 = (waypoints_[i] - waypoints_[(i - 1 + n) % n]).normalized();
            d2 = (waypoints_[(i + 1) % n] - waypoints_[i]).normalized();
        }

        // Left normals of each incident segment
        Vec2 n1 = d1.perpendicular();
        Vec2 n2 = d2.perpendicular();

        // Miter bisector: direction that preserves hw distance on both sides
        Vec2 miter = n1 + n2;
        float miterLen2 = miter.lengthSq();

        Vec2 offset;
        if (miterLen2 < 1e-6f) {
            // ~180° turn: normals cancel (hairpin). Fall back to outgoing normal.
            offset = n2 * hw;
        } else {
            miter = miter * (1.0f / std::sqrt(miterLen2));
            float dotVal = miter.dot(n1);
            if (dotVal < 1e-4f) {
                // Turn so sharp the miter flips sign — use outgoing normal
                offset = n2 * hw;
            } else {
                float scale = std::min(hw / dotVal, hw * miterLimit);
                offset = miter * scale;
            }
        }

        leftBorder_.push_back( waypoints_[i] + offset);
        rightBorder_.push_back(waypoints_[i] - offset);
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
    // Each track segment is a quad [L0, L1, R1, R0].
    // Decompose into 2 triangles — triangles are always convex, so this
    // handles non-convex quads (sharp curves) with zero gaps or false positives.
    int segs = (int)leftBorder_.size() - 1;
    for (int i = 0; i < segs; ++i) {
        Vec2 L0 = leftBorder_[i],  L1 = leftBorder_[i+1];
        Vec2 R0 = rightBorder_[i], R1 = rightBorder_[i+1];
        if (inTriangle(p, L0, L1, R0) || inTriangle(p, L1, R1, R0))
            return true;
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
