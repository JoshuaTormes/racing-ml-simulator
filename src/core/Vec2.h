#pragma once
#include <cmath>

struct Vec2 {
    float x = 0.f, y = 0.f;

    Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2 operator/(float s) const { return {x / s, y / s}; }
    Vec2& operator+=(Vec2 o) { x += o.x; y += o.y; return *this; }

    float dot(Vec2 o) const { return x * o.x + y * o.y; }
    float length() const { return std::sqrt(x * x + y * y); }
    float lengthSq() const { return x * x + y * y; }

    Vec2 normalized() const {
        float len = length();
        return len > 1e-9f ? Vec2{x / len, y / len} : Vec2{0.f, 0.f};
    }

    // Rotate by angle in radians (clockwise with y-down convention)
    Vec2 rotated(float rad) const {
        float c = std::cos(rad), s = std::sin(rad);
        return {x * c - y * s, x * s + y * c};
    }

    // Counter-clockwise perpendicular in y-down screen coords: {1,0} → {0,-1} (upward)
    Vec2 perpendicular() const { return {y, -x}; }
};

inline Vec2 operator*(float s, Vec2 v) { return {s * v.x, s * v.y}; }
