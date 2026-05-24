# Racing ML Sim

2D top-down racing game written in **C++17 + SFML 3** designed as a training environment for neural networks and neuroevolution.

## Dependencies

- macOS with [Homebrew](https://brew.sh)
- CMake ≥ 3.16
- SFML 3 (`brew install sfml`)
- `nlohmann/json` — fetched automatically by CMake at configure time

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run

### Windowed (keyboard control)

```bash
./build/racing_sim
# WASD or arrow keys to drive car 0
```

Options:
```
--map <path>          Map JSON (default: maps/map1.json)
--population <N>      Simultaneous cars (default: 1)
--seed <S>            RNG seed (default: 42)
--threads <K>         Worker threads (default: hardware_concurrency)
```

### Headless (ML training)

```bash
./build/racing_sim --headless --population 1000
```

### Benchmark

```bash
./build/racing_sim --benchmark --population 1000
# Prints wall-clock time for one full episode of 1000 cars @ 60 Hz
```

## Tests

```bash
ctest --test-dir build --output-on-failure
# Or directly:
cd build && ./racing_tests
```

Covers: Vec2, track geometry (raycast, borders, collision), sensor readings, neural network forward/serialization/error-rejection, determinism (single vs multi-thread), genetic algorithm, headless episode termination.

## Map Schema (`maps/*.json`)

```json
{
  "name": "map1",
  "closed": true,
  "track_width": 120.0,
  "waypoints": [[400,100],[700,200],...],
  "obstacles": [
    { "type": "circle", "pos": [400,350], "radius": 30 },
    { "type": "rect",   "pos": [550,250], "size": [40,80] }
  ]
}
```

- `closed`: `true` = lap circuit, `false` = point-to-point (finish at last waypoint).
- `track_width`: total width in world pixels; borders generated as perpendicular offset ±width/2.
- `waypoints`: ≥ 3 ordered `[x, y]` points defining the center-line.
- `obstacles`: optional list of static obstacles (`circle` or `rect`).

## Plugging in a new ML algorithm

1. **Implement `AIController`** in a new file:

```cpp
class MyRLController : public AIController {
public:
    Action decide(const Observation& obs) override {
        // obs has OBS_SIZE (10) floats:
        // [ray_0..ray_6, speed_norm, angle_to_next_wp, dist_to_next_wp]
        // Return Action{throttle ∈[-1,1], steering ∈[-1,1]}
        return {};
    }
    void reset() override { /* called at episode start */ }
};
```

2. **Single-agent RL loop** (Q-Learning, PPO, etc.):

```cpp
SimConfig cfg; cfg.population = 1;
Game game(cfg);
game.setControllers({ std::make_unique<MyRLController>() });
Observation obs = game.reset();
while (true) {
    Action act = myAgent.selectAction(obs);
    auto [next, reward, done] = game.step(act);
    myAgent.train(obs, act, reward, next, done);
    obs = next;
    if (done) obs = game.reset();
}
```

3. **Neuroevolution** (using the built-in `NeuralNetworkController` + `GeneticAlgorithm`):

```cpp
GeneticAlgorithm ga;
ga.initPopulation(1000, weightCount, seed);
for (int gen = 0; gen < maxGen; ++gen) {
    // Assign weights to controllers, run episode, collect fitness
    for (size_t i = 0; i < ga.genomes().size(); ++i) {
        auto& nn = dynamic_cast<NeuralNetworkController*>(game.controllers[i])->network();
        nn.setWeights(ga.genomes()[i].weights);
        ga.setFitness(i, game.cars()[i].fitness);
    }
    ga.evolve();
}
```

## Key constants (`src/core/Constants.h`)

| Constant | Value | Notes |
|---|---|---|
| `SIM_HZ` | 60 | Simulation steps per second |
| `DT` | 1/60 s | Fixed timestep |
| `NUM_RAYS` | 7 | Sensor rays (changing also changes `OBS_SIZE`) |
| `RAY_MAX_LEN` | 300 px | Sensor saturation distance |
| `MAX_SPEED` | 400 px/s | |
| `EPISODE_TIMEOUT` | 60 s | Max episode duration |
| `STALL_TIMEOUT` | 5 s | No-progress timeout |
| `OBS_SIZE` | 10 | Observation vector length |
