# Racing ML Sim

A 2D top-down racing game written in **C++17 + SFML 3**, designed as an **AI training environment via Machine Learning** — neuroevolution, Q-Learning, Policy Gradient, or any algorithm that implements the `reset() / step()` interface.

The core motivation is not the game itself, but the quality of the architecture as an RL environment: deterministic simulation, decoupled from rendering, runnable without a window (headless), and scalable to thousands of simultaneous agents.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation & Build](#installation--build)
3. [Execution Modes](#execution-modes)
4. [Controls (window mode)](#controls-window-mode)
5. [Command-line Options](#command-line-options)
6. [Tests](#tests)
7. [Creating New Maps](#creating-new-maps)
8. [Plugging in an ML Algorithm](#plugging-in-an-ml-algorithm)
9. [Configurable Constants](#configurable-constants)
10. [File Structure](#file-structure)

---

## Requirements

| Dependency | Minimum Version | How to Install |
|---|---|---|
| macOS | 13+ (Ventura) | — |
| Xcode Command Line Tools | 15+ | `xcode-select --install` |
| CMake | 3.16+ | `brew install cmake` |
| SFML | **3.x** | `brew install sfml` |
| nlohmann/json | 3.11.3 | Downloaded automatically by CMake |

> **Note:** the project uses the **SFML 3** API (events with `std::optional`, scoped enums, `sf::Text` with font in constructor). Not compatible with SFML 2.

---

## Installation & Build

```bash
# 1. Clone or navigate to the project directory
cd racing-ml-sim

# 2. Install dependencies (if not already installed)
brew install cmake sfml

# 3. Configure and compile in Release mode
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)

# Binary location:
./build/racing_sim
```

The build downloads `nlohmann/json` automatically via `FetchContent` on the first cmake run — internet access is required at this step.

---

## Execution Modes

### Window mode (default)

Opens a 900×700 window. **Car 0** (yellow) is controlled by the keyboard; the others have neural networks with random weights.

```bash
./build/racing_sim
./build/racing_sim --population 10   # more cars with random NNs
```

### Headless training (fastest — parallel)

Generational loop without a window. All **M maps × P cars** of a generation are evaluated in parallel using a worker thread pool — no map reloading between generations.

```bash
# 1000 cars, 5000 generations (recommended for serious training)
./build/racing_sim --train --headless --population 1000 --generations 5000

# Control thread count (default: all available cores)
./build/racing_sim --train --headless --population 1000 --generations 5000 --threads 8

# Control seed and output directory
./build/racing_sim --train --headless --seed 123 --out results/

# Resume from a saved champion
./build/racing_sim --train --headless --load out/best.rnnw --generations 5000
```

Terminal output (one line per generation):

```
gen    1/5000  agg= -0.308 | m0= 0.333 m1= 0.151 m2= 0.067 m3= 0.037 m4= 0.082 m5= 0.021 |  mean= -0.798  std= 0.242  done=0/1000  [col=480 stall=520 timeout=0]
gen    2/5000  agg=  0.412 | m0= 0.891 m1= 0.694 m2= 0.138 m3= 0.412 m4= 0.056 m5= 0.206 |  mean= -0.241  std= 0.381  done=8/1000  [col=320 stall=672 timeout=0]
...
```

Columns: `agg` = aggregated fitness (min across maps), `m0..m5` = best normalised fitness per map, `done` = cars that completed the circuit.

Files generated in `out/` (or `--out <dir>`):

| File | Content |
|---|---|
| `best.rnnw` | Weights of the **global champion** (overwritten when a new best is found) |
| `gen_0001.rnnw` … `gen_NNNN.rnnw` | Snapshot of the best individual in generation N |

### Training with window (serial — visualisation)

Same as headless, but opens a window to watch evolution live. **Note:** windowed mode is serial — maps are evaluated one at a time in sync with the renderer. Use `--headless` for large-scale training.

```bash
./build/racing_sim --train --population 100 --generations 200
```

With multiple maps, the window automatically cycles through the training maps within each generation — the current map name is shown on screen. Press **`T`** to toggle between **real-time** (60 Hz) and **turbo** (maximum speed).

### Watch a trained network

Loads a `.rnnw` file and opens a window with 1 car driving in a loop.

```bash
./build/racing_sim --watch out/best.rnnw
# or equivalently:
./build/racing_sim --load  out/best.rnnw
```

### Headless mode without training

Runs 1 complete episode and prints the elapsed time.

```bash
./build/racing_sim --headless --population 1000
```

### Benchmark mode

Measures simulation throughput and exits.

```bash
./build/racing_sim --benchmark --population 1000

# Example output:
# Benchmark: population=1000 threads=8
#   Wall-clock: 0.48s  (~3600000 car-ticks total)
```

```bash
# Compare single vs multi-thread
./build/racing_sim --benchmark --population 1000 --threads 1
./build/racing_sim --benchmark --population 1000 --threads 8
```

---

## Controls (window mode)

| Key | Mode | Action |
|---|---|---|
| `W` / `↑` | default | Accelerate |
| `S` / `↓` | default | Brake / reverse |
| `A` / `←` | default | Steer left |
| `D` / `→` | default | Steer right |
| `T` | `--train` | Toggle real-time ↔ turbo |
| Close window | all | Quit |

Car 0 (yellow) displays **sensor rays**: green = long distance, red = close to edge.

---

## Command-line Options

```
--headless              Run without a window
--map <path>            Path to the map JSON (default: maps/map1.json)
--population <N>        Number of simultaneous cars (default: 1)
--seed <S>              RNG seed — ensures reproducibility (default: 42)
--threads <K>           Threads for car updates (default: hardware_concurrency)
--benchmark             Measure throughput and exit

--train                 Enable the generational training loop
--algo <name>           Algorithm: genetic | random_search | hillclimb (default: genetic)
--generations <N>       Number of generations (default: 100)
--out <dir>             Output directory for weights (default: out/)
--load <file.rnnw>      With --train: seeds population from champion. Without --train: opens watch mode
--watch <file.rnnw>     Opens window with the saved network driving (no training)

--help / -h             Display help
```

**Mode precedence:** `--benchmark` > `--watch` > `--train` > `--load` (without train = watch) > default window mode.

---

## Tests

```bash
# Via CTest (run from the project root)
ctest --test-dir build --output-on-failure

# Directly (more verbose)
./build/racing_tests
```

Expected output:

```
=== Racing ML Sim Tests ===
Vec2: ok
Track geometry: ok
Sensor: ok
NeuralNetwork: ok
Determinism (single-thread): ok
Determinism (multi-thread): ok
GeneticAlgorithm: ok
Trainers weight count: ok
Trainers determinism: ok
Trainers elitism: ok
Trainers resume: ok
TrainingSession headless: ok
Game headless episode: ok

Results: 136 passed, 0 failed
```

### What the tests cover

| Group | What it verifies |
|---|---|
| Vec2 | length, normalized, rotated, dot, perpendicular |
| Track geometry | raycast with known distances, symmetric edges, `isInsideTrack`, JSON error handling |
| Sensor | readings ∈ [0,1], center ray reaches edge |
| NeuralNetwork | forward pass with known weights, round-trip save/load, rejection of wrong magic/version/topology |
| Determinism (single) | two instances with the same seed → bit-identical trajectory |
| Determinism (multi) | 100 cars, 300 ticks, single vs multi-thread → identical fitness per car |
| GeneticAlgorithm | initPopulation, setFitness, evolve increments generation |
| Trainers weight count | `weights(i).size() == weightCount` for all algorithms and individuals |
| Trainers determinism | two trainers with same seed + same fitness sequence → identical population after evolve |
| Trainers elitism | `random_search` and `hillclimb` maintain non-decreasing global best fitness |
| Trainers resume | `initFromWeights` preserves the champion in `weights(0)` |
| TrainingSession headless | 3-generation run generates `best.rnnw` + `gen_000N.rnnw` that can be reloaded |
| Headless episode | episode terminates and `episodeDone()` returns `true` |

---

## Creating New Maps

Maps are JSON files in `maps/`. Schema:

```json
{
  "name": "map_name",
  "closed": true,
  "track_width": 120.0,
  "waypoints": [
    [x0, y0],
    [x1, y1],
    ...
  ],
  "obstacles": [
    { "type": "circle", "pos": [cx, cy], "radius": 30.0 },
    { "type": "rect",   "pos": [cx, cy], "size": [width, height] }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `closed` | bool | yes | `true` = closed circuit (lap), `false` = open track (finish at last waypoint) |
| `track_width` | float | yes | Total track width in pixels. Edges generated with ±width/2 offset perpendicular to the centerline |
| `waypoints` | array of [x,y] | yes | Minimum 3 points. Defines the track **centerline** in order |
| `obstacles` | array | no | Static obstacles: `circle` (pos, radius) or `rect` (pos = center, size) |

**Coordinate convention:** origin at top-left, x grows right, y grows down. Angles in radians, clockwise direction.

**Spawn:** all cars start at `waypoints[0]`, oriented toward `waypoints[1]`.

Loading a different map:

```bash
./build/racing_sim --map maps/my_map.json
```

---

## Plugging in an ML Algorithm

### Option 1 — Single-agent RL (Q-Learning, PPO, SAC…)

Implement `AIController` and use the `reset()`/`step()` interface:

```cpp
#include "AIController.h"
#include "Game.h"

class MyAgent : public AIController {
public:
    Action decide(const Observation& obs) override {
        // obs[0..6]  → 7 normalized raycast readings [0,1]
        // obs[7]     → normalized speed [0,1]
        // obs[8]     → angle to next waypoint ∈ [-1,1]
        // obs[9]     → distance to next waypoint ∈ [0,1]
        //
        // Return Action{throttle, steering} both ∈ [-1, 1]
        return Action{1.f, 0.f}; // example: always accelerate straight
    }
    void reset() override { /* clear agent's internal state */ }
};

int main() {
    SimConfig cfg;
    cfg.population = 1;

    Game game(cfg);
    game.setControllers({ std::make_unique<MyAgent>() });

    Observation obs = game.reset();

    while (true) {
        // Agent chooses action
        Action act = myAgent.selectAction(obs);

        // Simulation advances 1 tick
        auto [next_obs, reward, done] = game.step(act);

        // Train agent with transition (obs, act, reward, next_obs, done)
        myAgent.train(obs, act, reward, next_obs, done);

        obs = next_obs;
        if (done) obs = game.reset();
    }
}
```

### Option 2 — Neuroevolution via CLI (recommended)

Use the command line directly — no code required:

```bash
# Headless training, 100 generations, 200 cars, GA
./build/racing_sim --train --headless --population 200 --generations 100 --out out/

# Watch the champion afterwards
./build/racing_sim --watch out/best.rnnw

# Continue training from the champion
./build/racing_sim --train --headless --load out/best.rnnw --generations 50
```

### Option 3 — Programmatic neuroevolution (TrainingSession)

To integrate the training loop into your own code:

```cpp
#include "Trainers.h"
#include "Training.h"

int main() {
    SimConfig cfg;
    cfg.population = 200;
    cfg.headless   = true;

    TrainingSession session(cfg, makeTrainer("genetic"), /*generations=*/100, "out/");
    session.runAll(); // prints per-generation stats and saves best.rnnw + gen_NNNN.rnnw

    // Inspect result
    const auto& stats = session.lastStats();
    std::cout << "Final best fitness: " << stats.bestFitness << "\n";
}
```

Or control tick by tick (to interleave with rendering):

```cpp
session.beginGeneration();
while (!session.generationComplete()) session.tick();
session.endGeneration(); // collects fitness, saves, evolves
```

### Option 4 — Manual neuroevolution (GeneticAlgorithm directly)

```cpp
#include "Game.h"
#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"

int main() {
    const int POP = 200, GENS = 100;
    SimConfig cfg; cfg.population = POP; cfg.headless = true;

    NeuralNetwork refNet({OBS_SIZE, 8, 2}, 0);
    GeneticAlgorithm ga;
    ga.initPopulation(POP, refNet.getWeights().size(), 42);

    for (int gen = 0; gen < GENS; ++gen) {
        std::vector<std::unique_ptr<AIController>> ctrls;
        for (int i = 0; i < POP; ++i) {
            NeuralNetwork nn({OBS_SIZE, 8, 2});
            nn.setWeights(ga.genomes()[i].weights);
            ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
        }
        Game game(cfg);
        game.setControllers(std::move(ctrls));
        game.runHeadlessEpisode();

        auto fits = game.fitnesses();
        for (int i = 0; i < POP; ++i) ga.setFitness(i, fits[i]);
        ga.evolve();
    }
}
```

### Option 5 — Fully custom algorithm

Implement the `Trainer` interface to use your own algorithm with the existing generational loop:

```cpp
#include "Trainer.h"

class MyCMAES : public Trainer {
public:
    void init(size_t popSize, size_t weightCount, unsigned seed) override { /* ... */ }
    void initFromWeights(const std::vector<float>& champion, size_t popSize, unsigned seed) override { /* ... */ }
    size_t populationSize() const override { return pop_.size(); }
    const std::vector<float>& weights(size_t i) const override { return pop_[i]; }
    void setFitness(size_t i, float f) override { fitness_[i] = f; }
    void evolve() override { /* CMA-ES update */ }
    int  generation() const override { return gen_; }
    const char* name() const override { return "cmaes"; }
private:
    std::vector<std::vector<float>> pop_;
    std::vector<float> fitness_;
    int gen_ = 0;
};

// Use with TrainingSession
TrainingSession session(cfg, std::make_unique<MyCMAES>(), 100, "out/");
session.runAll();
```

Or, for something even simpler, just implement `AIController::decide` and inject it via `game.setControllers(...)`. The `Game` doesn't know or care what's inside the controller.

---

## Configurable Constants

All located in `src/core/Constants.h`. Changing any of them requires recompiling.

| Constant | Default Value | Meaning |
|---|---|---|
| `SIM_HZ` | 60 | Simulation frequency (steps/second) |
| `DT` | 1/60 s | Fixed timestep per step |
| `NUM_RAYS` | 7 | Number of sensor rays (changes `OBS_SIZE` automatically) |
| `RAY_MAX_LEN` | 300 px | Maximum distance per ray; beyond that returns 1.0 |
| `MAX_SPEED` | 400 px/s | Maximum car speed |
| `ACCEL` | 300 px/s² | Acceleration with positive throttle |
| `BRAKE` | 500 px/s² | Deceleration/reverse with negative throttle |
| `DRAG` | 0.98 | Friction factor per tick (multiplies speed) |
| `MAX_STEER` | 3.0 rad/s | Maximum turn rate at maximum speed |
| `EPISODE_TIMEOUT` | 60 s | Maximum duration of an episode |
| `STALL_TIMEOUT` | 5 s | Maximum time without progress before `done` |
| `OBS_SIZE` | 10 | Fixed observation vector size (= `NUM_RAYS + 3`) |

Reward weights in `src/core/Types.h` (`RewardConfig`):

| Field | Default | Meaning |
|---|---|---|
| `w_progress` | 1.0 | Progress multiplier per tick |
| `w_speed` | 0.1 | Bonus for moving fast |
| `w_idle` | 0.05 | Penalty for staying still |
| `w_finish` | 100.0 | Bonus for completing the circuit |
| `w_crash` | 50.0 | Penalty for collision |
| `idle_eps` | 5 px/s | Speed threshold for idle penalty |

---

## File Structure

```
racing-ml-sim/
├── CMakeLists.txt          # Build: SFML 3 + nlohmann/json (FetchContent)
├── README.md               # Portuguese version
├── README-en.md            # This file (English version)
├── ARCHITECTURE.md         # Detailed technical documentation
├── assets/
│   └── DejaVuSans.ttf      # Open-source font for the HUD
├── maps/
│   └── map1.json           # Example circuit (8 waypoints, 2 obstacles)
├── src/
│   ├── core/
│   │   ├── Vec2.h          # 2D math vector, header-only, no SFML
│   │   ├── Constants.h     # All simulation constants
│   │   └── Types.h         # Observation, Action, StepResult, RewardConfig, SimConfig
│   ├── AIController.h      # Abstract interface: decide(Observation) → Action
│   ├── Track.h / .cpp      # Track: JSON, edges, raycast, progress
│   ├── Sensor.h / .cpp     # 7 normalized rays in a 180° fan
│   ├── Car.h / .cpp        # Physics, observation, reward, done conditions
│   ├── NeuralNetwork.h/.cpp# Feedforward MLP + binary RNNW serialization + NNController
│   ├── GeneticAlgorithm.h/.cpp # GA: init, seedFrom, evolve, crossover, mutation
│   ├── Trainer.h           # Trainer interface + GenerationStats struct (no SFML)
│   ├── Trainers.h / .cpp   # GeneticTrainer, RandomSearchTrainer, HillClimbTrainer + makeTrainer()
│   ├── Training.h / .cpp   # TrainingSession: generational loop, stats, save/load (no SFML)
│   ├── Game.h / .cpp       # reset/step (RL), batch tick, thread pool
│   ├── Renderer.h / .cpp   # ONLY SFML layer: track, cars, HUD, fitness chart
│   ├── HumanController.h/.cpp  # Keyboard input → Action (depends on SFML)
│   └── main.cpp            # Arg parsing, dispatch for all modes
└── tests/
    └── test_main.cpp       # 136 tests: Vec2, geometry, NN, determinism, GA, Trainers, TrainingSession
```