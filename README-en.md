# Racing ML Sim

A 2D top-down racing game written in **C++17 + SFML 3**, designed as an **AI training environment via Machine Learning** — neuroevolution, Q-Learning, Policy Gradient, or any algorithm that implements the `reset() / step()` interface.

The core motivation is not the game itself, but the quality of the architecture as an RL environment: deterministic simulation, decoupled from rendering, runnable without a window (headless), and scalable to thousands of simultaneous agents.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation & Build](#installation--build)
3. [Configuration File (train.json)](#configuration-file-trainjson)
4. [Execution Modes](#execution-modes)
5. [Controls (window mode)](#controls-window-mode)
6. [Command-line Options](#command-line-options)
7. [Tests](#tests)
8. [Creating New Maps](#creating-new-maps)
9. [Plugging in an ML Algorithm](#plugging-in-an-ml-algorithm)
10. [Configurable Constants](#configurable-constants)
11. [File Structure](#file-structure)

---

## Requirements

| Dependency | Minimum Version |
|---|---|
| C++17 compiler | clang 14+ / gcc 11+ / MSVC 2022 |
| CMake | 3.16+ |
| SFML | **3.x** |
| nlohmann/json | 3.11.3 (downloaded automatically by CMake) |

> **Note:** the project uses the **SFML 3** API (events with `std::optional`, scoped enums, `sf::Text` with font in constructor). Not compatible with SFML 2.

The project runs on **macOS, Linux, and Windows** — there are no platform-specific dependencies in the code.

---

## Installation & Build

### macOS

```bash
xcode-select --install
brew install cmake sfml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)
./build/racing_sim
```

### Linux

**Ubuntu 24.04+ / Debian 13+**
```bash
sudo apt install build-essential cmake libsfml-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/racing_sim
```

**Arch Linux**
```bash
sudo pacman -S cmake sfml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/racing_sim
```

**Fedora**
```bash
sudo dnf install cmake gcc-c++ SFML-devel

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/racing_sim
```

> On older distros (e.g. Ubuntu 22.04), `libsfml-dev` may install SFML 2. Check with `apt show libsfml-dev | grep Version`. If it's 2.x, either [build SFML 3 from source](https://github.com/SFML/SFML) or use a newer distro.

### Windows

**Visual Studio 2022 + vcpkg**
```powershell
# Install Visual Studio 2022 with the "Desktop development with C++" workload
# Install vcpkg: https://github.com/microsoft/vcpkg
vcpkg install sfml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
# Binary: build\Release\racing_sim.exe
```

**MSYS2 / MinGW-w64**
```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-sfml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
cmake --build build -j$(nproc)
./build/racing_sim.exe
```

---

The build downloads `nlohmann/json` automatically via `FetchContent` on the first cmake run — internet access is required at this step.

---

## Configuration File (train.json)

`train.json` is the central experiment configuration file. Set it up correctly before training — the simulator auto-detects and loads it on startup.

```json
{
  "population": 1000,
  "generations": 1500,
  "threads": 8,
  "seed": 42,
  "headless": true,
  "out": "out_v2",
  "log_csv": true,

  "train_maps": [
    "maps/map1_chicanes_infernais.json",
    "maps/map4_obstaculos.json",
    "maps/map5_tecnico_avancado.json"
  ],
  "val_maps": ["maps/map7_pesadelo.json"],
  "test_maps": ["maps/map8_caos_total.json"],

  "fitness_agg": "cvar-rank",
  "cvar_alpha": 1.0,
  "curriculum": "none",

  "augment": ["mirror", "reverse"],
  "procedural_train": 4,
  "proc_width_min": 60,
  "proc_width_max": 80,

  "random_spawn": true,
  "episodes_per_eval": 2,

  "select_by_val": true,
  "val_select_topk": 10
}
```

JSON keys use underscores and mirror the CLI flags: `train_maps`, `cvar_alpha`, `random_spawn`, `episodes_per_eval`, etc. Arrays are accepted for `train_maps`, `val_maps`, `test_maps`, `augment`, and `map_weights`.

To use an alternate config (e.g. parallel experiments):

```bash
./build/racing_sim --train --config experiment_b.json
```

---

## Execution Modes

### Training

With `train.json` configured, run:

```bash
./build/racing_sim --train
```

With `headless: true` in `train.json`, training runs without a window — much faster for large populations, since all **M maps × P cars** are evaluated in parallel using a worker thread pool.

Terminal output (one line per generation):

```
gen    1/1500  agg= -0.308 | m0= 0.333 m1= 0.151 m2= 0.067 |  mean= -0.798  std= 0.242  done=0/1000  [col=480 stall=520 timeout=0]
gen    2/1500  agg=  0.412 | m0= 0.891 m1= 0.694 m2= 0.138 |  mean= -0.241  std= 0.381  done=8/1000  [col=320 stall=672 timeout=0]
...
```

Columns: `agg` = aggregated fitness (mode configurable via `--fitness-agg`), `m0..mN` = best normalised fitness per map, `done` = cars that completed the circuit.

Files generated in the directory set by `out`:

| File | Content |
|---|---|
| `best.rnnw` | Weights of the **global champion** (overwritten when a new best is found) |
| `gen_0001.rnnw` … `gen_NNNN.rnnw` | Snapshot of the best individual in generation N |
| `training_log.csv` | Per-generation training metrics (requires `--log-csv`) |
| `held_out_log.csv` | Per-generation validation metrics (requires `--log-csv`) |
| `test_log.csv` | Test-set metrics (requires `--log-csv` + `test_maps`) |

### Resuming a training run

To continue from a saved checkpoint, use `--load` pointing to the desired `.rnnw`. The simulator uses the champion's weights to seed the initial population and resumes training from there:

```bash
./build/racing_sim --train --load out_v2/best.rnnw
```

All parameters (maps, population, generations, etc.) still come from `train.json`. To run more generations than configured:

```bash
./build/racing_sim --train --load out_v2/best.rnnw --generations 2000
```

### Training with window (visualisation)

When `headless: false` in `train.json` (or without `--headless`), a window opens to watch evolution live. **Note:** windowed mode is serial — maps are evaluated one at a time in sync with the renderer.

```bash
./build/racing_sim --train
```

With multiple maps, the window automatically cycles through the training maps. Press **`T`** to toggle between **real-time** (60 Hz) and **turbo** (maximum speed).

### Window mode (explore / play)

Opens a 900×700 window. **Car 0** (yellow) is controlled by the keyboard; the others have neural networks with random weights.

```bash
./build/racing_sim
./build/racing_sim --population 10   # more cars with random NNs
```

### Watch a trained network

Loads a `.rnnw` file and opens a window with 1 car driving in a loop.

```bash
./build/racing_sim --watch out_v2/best.rnnw
# or equivalently:
./build/racing_sim --load out_v2/best.rnnw
```

### Human vs AI

You control the yellow car; the saved network drives the green one(s). Use `←→↑↓` or `WASD`.

```bash
./build/racing_sim --versus out_v2/best.rnnw

# Multiple AIs (--population N), with weight noise so they diverge from each other:
./build/racing_sim --versus out_v2/best.rnnw --population 5
./build/racing_sim --versus out_v2/best.rnnw --population 5 --versus-noise 0.05

# On a specific map:
./build/racing_sim --versus out_v2/best.rnnw --map maps/map3.json
```

The race restarts automatically when everyone finishes. Press **Restart** (or `R`) to restart at any time.

### Headless mode without training

Runs 1 complete episode and prints the elapsed time.

```bash
./build/racing_sim --headless --population 1000
```

### Benchmark mode

Measures simulation throughput and exits.

```bash
./build/racing_sim --benchmark --population 1000

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

Complete reference for all available flags. Normal usage is to configure experiments in `train.json` — the flags below are for one-off overrides and special modes.

```
Config file
  --config <file.json>  Load defaults from a JSON file (CLI flags override)
  (auto-detect)         train.json in the current directory is loaded automatically

Basic
  --headless              Run without a window
  --map <path>            Path to the map JSON (default: maps/map1_chicanes_infernais.json)
  --population <N>        Number of simultaneous cars (default: 1)
  --seed <S>              RNG seed — same seed → identical training run (default: 42)
  --threads <K>           Threads for car updates (default: hardware_concurrency)
  --benchmark             Measure throughput and exit

Training
  --train                 Enable the generational training loop
  --algo <name>           Algorithm: genetic | random_search | hillclimb (default: genetic)
  --generations <N>       Number of generations (default: 100)
  --out <dir>             Output directory for weights (default: out/)
  --load <file.rnnw>      With --train: seeds population from champion. Without --train: opens watch mode
  --log-csv               Save training_log.csv, held_out_log.csv (and test_log.csv if --test-maps) to <out>
  --hidden <N>            Hidden layer neurons (default: 32)
  --episode-timeout <s>   Max episode duration in seconds (default: 30)

Multi-map generalization
  --train-maps <a,b,...>  Comma-separated list of training maps
  --val-maps <x,y>        Validation maps (used for selection when --select-by-val)
  --test-maps <x,y>       Test maps: report only (test_log.csv), never used for selection
  --fitness-agg <mode>    Aggregate fitness across maps: cvar-rank (default) | min | mean | cvar-raw
  --cvar-alpha <α>        CVaR tail fraction ∈ (0,1] (default: 0.5). Used with cvar-rank/cvar-raw
  --map-norm <mode>       Per-map normalization: zscore (default) | minmax | progress
                          Ignored under cvar-rank (ranks are scale-invariant)
  --map-weights <w,...>   Per-map weights for cvar-rank (one per train map; default: all 1.0)
  --progressive-frac <f>  Fraction of population evaluated on maps beyond the first (default: 1.0)
  --finetune-map <path>   Replace map[0] with this JSON — useful to specialize on a new track

Curriculum (off by default)
  --curriculum <mode>     linear (default) | none | explicit
  --curriculum-start <N>  Generation when the second map is introduced (linear, default: 2)
  --curriculum-step <N>   Generations between each subsequent map addition (linear, default: 15)
  --curriculum-schedule <g,...>  Generation thresholds for explicit mode (M−1 values)
  --curriculum-pin <i,...>       Map indices always active regardless of curriculum

Generalization (anti-overfitting — all off by default)
  --augment <list>        Extra train maps: mirror,reverse,width:0.85,width:1.15
  --procedural-train <K>  Generate K seeded random tracks into the train set
  --procedural-val <K>    Generate K seeded random tracks into the validation set
  --proc-width-min <w>    Min width for procedural tracks (default 55)
  --proc-width-max <w>    Max width for procedural tracks (default 110)
  --dump-gen-maps <dir>   Save augmented+procedural maps as JSON to <dir> before training
  --random-spawn          Random start point per training episode (progress measured from spawn)
  --sensor-noise <s>      Gaussian noise (stddev) on ray readings during training
  --episodes-per-eval <N> Episodes per (genome,map), aggregated (default 1)
  --episode-agg <mode>    Combine episodes: mean (default) | min
  --select-by-val         Save best.rnnw by validation progress (top-K) instead of train fitness
  --val-select-topk <T>   Train genomes evaluated on validation for selection (default 1)

Watch / Interactive
  --watch <file.rnnw>     Opens window with the saved network driving (no training)
  --versus <file.rnnw>    Human (yellow, WASD/arrows) vs AI (green) with saved network

  --help / -h             Display help
```

**Mode precedence:** `--benchmark` > `--watch` > `--versus` > `--train` > `--load` (without train = watch) > default window mode.

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
        // obs[0..12] → 13 normalized raycast readings [0,1] (180° fan)
        // obs[13]    → normalized speed |v|/MAX_SPEED ∈ [0,1]
        // obs[14]    → lateral offset from centerline ∈ [-1,1] (positive = right)
        // obs[15]    → heading error vs track tangent ∈ [-1,1]
        // obs[16,17] → lookahead 1: (signed_curvature ∈ [-1,1], speed_excess ∈ [-1,1])
        // obs[18,19] → lookahead 2
        // obs[20,21] → lookahead 3
        // obs[22,23] → lookahead 4
        // obs[24,25] → lookahead 5 (furthest)
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
| `NUM_RAYS` | 13 | Number of sensor rays (changes `OBS_SIZE` automatically) |
| `RAY_MAX_LEN` | 400 px | Maximum distance per ray; beyond that returns 1.0 |
| `MAX_SPEED` | 400 px/s | Maximum car speed |
| `MAX_REVERSE_SPEED` | 0 px/s | Maximum reverse speed (0 = reverse disabled) |
| `ACCEL` | 300 px/s² | Acceleration with positive throttle |
| `BRAKE` | 500 px/s² | Deceleration with negative throttle |
| `DRAG` | 0.98 | Friction factor per tick (0.98^60 ≈ 0.30 in 1s) |
| `MAX_STEER` | 3.0 rad/s | Maximum turn rate |
| `MAX_LAT_ACCEL` | 650 px/s² | Lateral grip limit (yawRate ≤ MAX_LAT_ACCEL/v) |
| `EPISODE_TIMEOUT` | 30 s | Maximum episode duration (overridable via `--episode-timeout`) |
| `STALL_TIMEOUT` | 2 s | Time without minimum arc progress before `done` |
| `STALL_SPEED` | 4 px/s | Speed below which the stall timer runs |
| `STALL_PROGRESS_MIN` | 0.003 | Min arc progress fraction to reset the stall timer |
| `NUM_LOOKAHEADS` | 5 | Number of curvature lookahead points in the observation |
| `OBS_SIZE` | 26 | Observation vector size (= `NUM_RAYS + 3 + 2 × NUM_LOOKAHEADS`) |
| `NN_HIDDEN` | 32 | Hidden layer neurons (overridable via `--hidden`) |

Reward weights in `src/core/Types.h` (`RewardConfig`):

| Field | Default | Meaning |
|---|---|---|
| `w_progress` | 200.0 | Weight on max arc progress ∈ [0,1] |
| `w_speed` | 0.3 | Bonus per unit of speed while advancing |
| `w_checkpoint` | 5.0 | One-shot bonus per design waypoint crossed, scaled by curvature |
| `w_finish` | 100.0 | Bonus for completing the circuit |
| `w_time` | 2.0 | Time-remaining bonus on completion (`w_time × (timeout − t)`) |
| `w_crash` | 15.0 | Penalty for collision |
| `w_reverse` | 0.5 | Accumulated penalty for reverse speed (per unit/s) |
| `w_regress` | 2.0 | Penalty for regressing behind peak progress (per unit/s) |
| `w_curve` | 0.0 | Penalty for high speed in tight corners (disabled by default) |

---

## File Structure

```
racing-ml-sim/
├── CMakeLists.txt          # Build: SFML 3 + nlohmann/json (FetchContent)
├── README.md               # Portuguese version
├── README-en.md            # This file (English version)
├── ARCHITECTURE.md         # Detailed technical documentation
├── train.json              # Default experiment config (auto-loaded by the simulator)
├── assets/
│   └── DejaVuSans.ttf      # Open-source font for the HUD
├── maps/
│   ├── map1_chicanes_infernais.json  # Track with chicane sequence
│   ├── map4_obstaculos.json          # Track with static obstacles
│   ├── map5_tecnico_avancado.json    # Advanced technical track
│   ├── map7_pesadelo.json            # Validation map
│   └── map8_caos_total.json          # Test map (held-out)
├── src/
│   ├── core/
│   │   ├── Vec2.h          # 2D math vector, header-only, no SFML
│   │   ├── Constants.h     # All simulation constants
│   │   ├── Types.h         # Observation, Action, StepResult, RewardConfig, SimConfig
│   │   └── TrackGen.h/.cpp # Procedural track generation and augmentation (mirrorX, reverse, scaleWidth)
│   ├── AIController.h      # Abstract interface: decide(Observation) → Action
│   ├── Track.h / .cpp      # Track: JSON/in-memory, Catmull-Rom edges, raycast, arc-length progress
│   ├── Sensor.h / .cpp     # 13 normalized rays in a 180° fan
│   ├── Car.h / .cpp        # Physics, observation (26 floats), reward, done conditions
│   ├── NeuralNetwork.h/.cpp# Feedforward MLP + binary RNNW serialization + NNController
│   ├── GeneticAlgorithm.h/.cpp # GA: init, seedFrom, evolve, crossover, mutation
│   ├── Trainer.h           # Trainer interface + GenerationStats struct (no SFML)
│   ├── Trainers.h / .cpp   # GeneticTrainer, RandomSearchTrainer, HillClimbTrainer + makeTrainer()
│   ├── Training.h / .cpp   # TrainingSession: multi-map loop, curriculum, augmentation, val/test
│   ├── TrainingMath.h/.cpp # CVaR, rank-CVaR, z-score and fitness aggregations
│   ├── Game.h / .cpp       # reset/step (RL), batch tick, thread pool
│   ├── Renderer.h / .cpp   # ONLY SFML layer: track, cars, HUD, fitness chart
│   ├── HumanController.h/.cpp  # Keyboard input → Action (depends on SFML)
│   └── main.cpp            # Arg parsing, dispatch for all modes
├── tools/
│   ├── check_map_overlap.py  # Detect track ribbon self-overlap; optional PNG output
│   └── watch_training.py     # Live-plot fitness/validation curves during training
└── tests/
    └── test_main.cpp       # Tests: Vec2, geometry, NN, determinism, GA, Trainers, TrainingSession, reward
```