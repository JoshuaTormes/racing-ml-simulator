#ifndef HEADLESS_ONLY
#include "render/Renderer.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cstdlib>

Renderer::Renderer(unsigned width, unsigned height, const std::string& fontPath)
    : window_(sf::VideoMode({width, height}), "Racing ML Sim")
    , speedFieldRect_(sf::Vector2f{440.f, 8.f}, sf::Vector2f{156.f, 26.f})
    , prevMapBtn_(sf::Vector2f{606.f, 8.f}, sf::Vector2f{90.f, 26.f})
    , restartBtn_(sf::Vector2f{704.f, 8.f}, sf::Vector2f{90.f, 26.f})
    , nextMapBtn_(sf::Vector2f{802.f, 8.f}, sf::Vector2f{90.f, 26.f})
{
    window_.setFramerateLimit(60);
    fontLoaded_ = font_.openFromFile(fontPath);
}

bool Renderer::handleEvents() {
    while (auto e = window_.pollEvent()) {
        if (e->is<sf::Event::Closed>()) {
            window_.close();
            return false;
        }
        if (const auto* kp = e->getIf<sf::Event::KeyPressed>()) {
            if (speedFocused_) {
                if (kp->code == sf::Keyboard::Key::Enter)
                    commitSpeed();
                else if (kp->code == sf::Keyboard::Key::Escape)
                    speedFocused_ = false; // cancel, keep previous value
            }
        }
        if (speedFocused_) {
            if (const auto* te = e->getIf<sf::Event::TextEntered>()) {
                char32_t u = te->unicode;
                if (u >= U'0' && u <= U'9') {
                    if (speedText_.size() < 5) speedText_ += (char)u;
                } else if (u == 8 && !speedText_.empty()) { // backspace
                    speedText_.pop_back();
                }
            }
        }
        if (const auto* mb = e->getIf<sf::Event::MouseButtonPressed>()) {
            if (mb->button == sf::Mouse::Button::Left) {
                sf::Vector2f p((float)mb->position.x, (float)mb->position.y);
                if (speedFieldRect_.contains(p)) {
                    speedFocused_ = true;
                    speedText_.clear();
                } else {
                    if (speedFocused_) commitSpeed(); // click away commits
                    if (restartBtn_.contains(p))
                        restartClicked_ = true;
                    else if (prevMapBtn_.contains(p))
                        mapDelta_ = -1;
                    else if (nextMapBtn_.contains(p))
                        mapDelta_ = +1;
                }
            }
        }
    }
    return window_.isOpen();
}

void Renderer::commitSpeed() {
    if (!speedText_.empty()) {
        int v = std::atoi(speedText_.c_str());
        if (v < 1) v = 1;
        if (v > 100000) v = 100000;
        speedValue_ = (float)v;
    }
    speedFocused_ = false;
    speedText_.clear();
}

bool Renderer::consumeRestart() {
    bool v = restartClicked_;
    restartClicked_ = false;
    return v;
}

int Renderer::consumeMapDelta() {
    int v = mapDelta_;
    mapDelta_ = 0;
    return v;
}

void Renderer::drawTrack(const Track& track) {
    auto drawPoly = [&](const std::vector<Vec2>& pts, sf::Color col) {
        int n = (int)pts.size();
        for (int i = 0; i + 1 < n; ++i) {
            sf::Vertex line[2] = {
                sf::Vertex{toSf(pts[i]),   col},
                sf::Vertex{toSf(pts[i+1]), col}
            };
            window_.draw(line, 2, sf::PrimitiveType::Lines);
        }
    };
    // Dense centerline (thin gray) — drawn before borders so they sit on top.
    {
        const auto& center = track.centerline();
        for (int i = 0; i + 1 < (int)center.size(); ++i) {
            sf::Vertex line[2] = {
                sf::Vertex{toSf(center[i]),     sf::Color(80, 80, 80, 120)},
                sf::Vertex{toSf(center[i + 1]), sf::Color(80, 80, 80, 120)}
            };
            window_.draw(line, 2, sf::PrimitiveType::Lines);
        }
    }

    drawPoly(track.leftBorder(),  sf::Color::White);
    drawPoly(track.rightBorder(), sf::Color::White);

    // Design waypoints (sparse anchors from JSON) as small blue dots.
    for (auto& wp : track.designWaypoints()) {
        sf::CircleShape dot(4.f);
        dot.setFillColor(sf::Color(100, 100, 255, 120));
        dot.setPosition({wp.x - 4.f, wp.y - 4.f});
        window_.draw(dot);
    }

    // Start/finish lines use border points indexed by design waypoints.
    // Each designWaypoint k corresponds to centerline_[k * CENTERLINE_SUBSEGMENTS],
    // which is also the index into leftBorder_ / rightBorder_.
    const auto& dwps = track.designWaypoints();
    const auto& lb   = track.leftBorder();
    const auto& rb   = track.rightBorder();
    int nDesign = (int)dwps.size();

    // Draw a solid colored line
    auto drawLine = [&](Vec2 a, Vec2 b, sf::Color col, float thickness) {
        Vec2 d = {b.x - a.x, b.y - a.y};
        float len = std::sqrt(d.x*d.x + d.y*d.y);
        if (len < 1.f) return;
        sf::RectangleShape rect({len, thickness});
        rect.setFillColor(col);
        rect.setOrigin({0.f, thickness * 0.5f});
        rect.setPosition({a.x, a.y});
        rect.setRotation(sf::radians(std::atan2(d.y, d.x)));
        window_.draw(rect);
    };

    // Draw a checkered line (alternating black/white tiles)
    auto drawCheckered = [&](Vec2 a, Vec2 b, float thickness) {
        Vec2 d = {b.x - a.x, b.y - a.y};
        float len = std::sqrt(d.x*d.x + d.y*d.y);
        if (len < 1.f) return;
        int tiles = 8;
        for (int t = 0; t < tiles; ++t) {
            float t0 = (float)t / tiles;
            Vec2 p = {a.x + d.x * t0, a.y + d.y * t0};
            sf::RectangleShape rect({len / tiles, thickness});
            rect.setFillColor((t % 2 == 0) ? sf::Color::White : sf::Color::Black);
            rect.setOrigin({0.f, thickness * 0.5f});
            rect.setPosition({p.x, p.y});
            rect.setRotation(sf::radians(std::atan2(d.y, d.x)));
            window_.draw(rect);
        }
    };

    // Start line — border points at the spawn design waypoint (index 0).
    if (!lb.empty()) {
        drawLine(lb[0], rb[0], sf::Color(255, 255, 255, 200), 5.f);
    }

    // Finish line — border points at the last design waypoint.
    if (nDesign >= 2) {
        int finishIdx = (nDesign - 1) * CENTERLINE_SUBSEGMENTS;
        if (finishIdx < (int)lb.size())
            drawCheckered(lb[finishIdx], rb[finishIdx], 6.f);
    }

    // Obstacles
    for (auto& ob : track.obstacles()) {
        if (ob.type == Obstacle::Type::Circle) {
            sf::CircleShape c(ob.radius);
            c.setFillColor(sf::Color(180, 60, 60, 200));
            c.setPosition({ob.pos.x - ob.radius, ob.pos.y - ob.radius});
            window_.draw(c);
        } else {
            sf::RectangleShape r({ob.size.x, ob.size.y});
            r.setFillColor(sf::Color(180, 60, 60, 200));
            r.setPosition({ob.pos.x - ob.size.x * 0.5f, ob.pos.y - ob.size.y * 0.5f});
            window_.draw(r);
        }
    }
}

void Renderer::drawCar(const Car& car, sf::Color color) {
    sf::RectangleShape rect({20.f, 10.f});
    rect.setFillColor(color);
    rect.setOrigin({10.f, 5.f});
    rect.setPosition(toSf(car.pos));
    rect.setRotation(sf::radians(car.angle));
    window_.draw(rect);
}

void Renderer::drawRays(const Car& car) {
    auto& hits = car.sensor.hitPoints();
    auto& readings = car.sensor.readings();
    for (int i = 0; i < NUM_RAYS; ++i) {
        // Color: green = far, red = near
        float t = readings[i];
        sf::Color col(
            (uint8_t)(255 * (1.f - t)),
            (uint8_t)(255 * t),
            0, 140);
        sf::Vertex line[2] = {
            sf::Vertex{toSf(car.pos),    col},
            sf::Vertex{toSf(hits[i]),    col}
        };
        window_.draw(line, 2, sf::PrimitiveType::Lines);
    }
}

void Renderer::drawHUD(const Game& game) {
    if (!fontLoaded_ || game.cars().empty()) return;
    const Car& c = game.cars()[0];

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "Speed: " << (int)std::fabs(c.speed) << " px/s\n";
    ss << "Time:  " << c.episodeTime << " s\n";
    ss << "Prog:  " << std::setprecision(2) << c.maxProgress << "\n";
    ss << "Cars:  " << game.config().population;

    sf::Text text(font_, ss.str(), 16);
    text.setFillColor(sf::Color::White);
    text.setPosition({8.f, 8.f});
    window_.draw(text);
}

void Renderer::drawButton(const sf::FloatRect& r, const std::string& label) {
    sf::RectangleShape rect({r.size.x, r.size.y});
    rect.setPosition({r.position.x, r.position.y});
    rect.setFillColor(sf::Color(60, 60, 70));
    rect.setOutlineColor(sf::Color(160, 160, 160));
    rect.setOutlineThickness(1.f);
    window_.draw(rect);

    if (fontLoaded_) {
        sf::Text text(font_, label, 13);
        text.setFillColor(sf::Color::White);
        auto bounds = text.getLocalBounds();
        float tx = r.position.x + (r.size.x - bounds.size.x) * 0.5f - bounds.position.x;
        float ty = r.position.y + (r.size.y - bounds.size.y) * 0.5f - bounds.position.y;
        text.setPosition({tx, ty});
        window_.draw(text);
    }
}

void Renderer::drawControls(const std::string& mapName) {
    drawButton(prevMapBtn_, "< Mapa");
    drawButton(restartBtn_, "Restart");
    drawButton(nextMapBtn_, "Mapa >");

    if (fontLoaded_) {
        sf::Text lbl(font_, mapName, 13);
        lbl.setFillColor(sf::Color(200, 200, 200));
        auto bounds = lbl.getLocalBounds();
        float x = nextMapBtn_.position.x + nextMapBtn_.size.x - bounds.size.x - bounds.position.x;
        lbl.setPosition({x, 40.f});
        window_.draw(lbl);
    }
}

void Renderer::drawSpeedField() {
    const auto& r = speedFieldRect_;
    sf::RectangleShape box({r.size.x, r.size.y});
    box.setPosition({r.position.x, r.position.y});
    box.setFillColor(speedFocused_ ? sf::Color(40, 60, 90) : sf::Color(60, 60, 70));
    box.setOutlineColor(speedFocused_ ? sf::Color(120, 200, 255) : sf::Color(160, 160, 160));
    box.setOutlineThickness(speedFocused_ ? 2.f : 1.f);
    window_.draw(box);

    if (fontLoaded_) {
        std::ostringstream ss;
        ss << "Vel: ";
        if (speedFocused_)
            ss << (speedText_.empty() ? "" : speedText_) << "_";
        else
            ss << (int)speedValue_ << "x";

        sf::Text text(font_, ss.str(), 14);
        text.setFillColor(sf::Color::White);
        text.setPosition({r.position.x + 8.f, r.position.y + 5.f});
        window_.draw(text);
    }
}

void Renderer::render(const Game& game, bool showRays) {
    window_.clear(sf::Color(30, 30, 30));
    drawTrack(game.track());

    const auto& cars = game.cars();
    int total = (int)cars.size();
    int displayLimit = std::min(total, 200);
    for (int i = 0; i < total; ++i) {
        if (cars[i].done) {
            if (i < displayLimit && cars[i].doneReason == DoneReason::Completed)
                drawCar(cars[i], sf::Color(0, 255, 120, 200)); // green = completed
            continue;
        }
        sf::Color col = (i == 0) ? sf::Color::Yellow : sf::Color(100, 200, 100, 180);
        drawCar(cars[i], col);
        if (showRays && i == 0) drawRays(cars[i]);
    }
    drawHUD(game);
    drawControls(game.track().name());
    window_.display();
}

void Renderer::renderTraining(const TrainingSession& session) {
    window_.clear(sf::Color(30, 30, 30));
    drawTrack(session.game().track());

    const auto& cars = session.game().cars();
    int total = (int)cars.size();
    int displayLimit = std::min(total, 200); // cap completed/dead cars for performance

    // Search ALL cars for the alive car furthest ahead (fixes invisible cars beyond index 200)
    int bestIdx = -1;
    float bestP = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < total; ++i) {
        if (!cars[i].done && cars[i].maxProgress > bestP) {
            bestP = cars[i].maxProgress;
            bestIdx = i;
        }
    }

    for (int i = 0; i < total; ++i) {
        if (cars[i].done) {
            // Only draw completed cars within display limit (performance)
            if (i < displayLimit && cars[i].doneReason == DoneReason::Completed)
                drawCar(cars[i], sf::Color(0, 255, 120, 200)); // green = completed lap
            continue;
        }
        // Always draw alive cars regardless of index
        sf::Color col = (i == bestIdx) ? sf::Color::Yellow : sf::Color(100, 200, 100, 180);
        drawCar(cars[i], col);
    }

    drawTrainingHUD(session);
    if (bestIdx >= 0) drawCarDebugHUD(cars[bestIdx]);
    drawFitnessGraph(session.history());
    drawControls(session.game().track().name());
    drawSpeedField();
    window_.display();
}

void Renderer::drawTrainingHUD(const TrainingSession& session) {
    if (!fontLoaded_) return;

    const auto& stats = session.lastStats();
    const auto& cars  = session.game().cars();
    int total = (int)cars.size();
    int elim  = (int)std::count_if(cars.begin(), cars.end(), [](const Car& c){ return c.done; });
    int alive = total - elim;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Gen: " << session.currentGeneration() + 1 << "/" << session.totalGenerations() << "\n";
    ss << "Algo: " << session.algoName() << "\n";
    ss << "Total: " << total << "  Vivos: " << alive << "  Elim: " << elim << "\n";
    if (stats.generation > 0) {
        ss << "Best: " << stats.aggBest << "\n";
        ss << "Mean: " << stats.aggMean << "\n";
        int totalDone = 0;
        for (const auto& pm : stats.perMap)
            if (pm.active) totalDone += pm.nCompleted;
        ss << "Done: " << totalDone << "/" << stats.population << "\n";
    }
    ss << "Velocidade: " << (int)speedValue_ << "x";

    sf::Text text(font_, ss.str(), 16);
    text.setFillColor(sf::Color::White);
    text.setPosition({8.f, 8.f});
    window_.draw(text);
}

void Renderer::drawCarDebugHUD(const Car& car) {
    if (!fontLoaded_) return;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "spd: "       << std::fabs(car.speed) << " px/s\n";
    ss << "cp: "        << car.lastCheckpoint    << "\n";
    ss << "noProgress: "<< car.noProgressTime   << "s\n";
    ss << "lowSpd: "    << car.lowSpeedTime      << "s\n";
    ss << "fitness: "   << car.fitness;

    sf::Text text(font_, ss.str(), 13);
    text.setFillColor(sf::Color(255, 220, 80));
    // Position near bottom-left, above the fitness graph area
    auto sz = window_.getSize();
    text.setPosition({10.f, (float)sz.y - 170.f});
    window_.draw(text);
}

void Renderer::drawFitnessGraph(const std::vector<GenerationStats>& history) {
    if (history.size() < 2) return;

    // Panel in bottom-right: 200x100 px, 10px margin
    auto sz = window_.getSize();
    float pw = 200.f, ph = 100.f, margin = 10.f;
    float px = (float)sz.x - pw - margin;
    float py = (float)sz.y - ph - margin;

    // Background
    sf::RectangleShape bg({pw, ph});
    bg.setPosition({px, py});
    bg.setFillColor(sf::Color(0, 0, 0, 160));
    bg.setOutlineColor(sf::Color(120, 120, 120));
    bg.setOutlineThickness(1.f);
    window_.draw(bg);

    // Find min/max fitness
    float minF =  std::numeric_limits<float>::infinity();
    float maxF = -std::numeric_limits<float>::infinity();
    for (const auto& s : history) {
        if (s.aggBest < minF) minF = s.aggBest;
        if (s.aggBest > maxF) maxF = s.aggBest;
    }
    float range = (maxF - minF > 1e-6f) ? maxF - minF : 1.f;

    // Draw polyline
    std::vector<sf::Vertex> pts;
    pts.reserve(history.size());
    for (size_t i = 0; i < history.size(); ++i) {
        float t = (float)i / (float)(history.size() - 1);
        float norm = (history[i].aggBest - minF) / range;
        float x = px + t * pw;
        float y = py + ph - norm * ph;
        pts.push_back(sf::Vertex{{x, y}, sf::Color(100, 220, 100)});
    }
    window_.draw(pts.data(), pts.size(), sf::PrimitiveType::LineStrip);

    // Label
    if (fontLoaded_) {
        sf::Text lbl(font_, "Best Fitness", 11);
        lbl.setFillColor(sf::Color(180, 180, 180));
        lbl.setPosition({px + 4.f, py + 2.f});
        window_.draw(lbl);
    }
}

#endif // HEADLESS_ONLY
