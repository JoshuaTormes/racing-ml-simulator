# Racing ML Sim

Jogo 2D de corrida top-down escrito em **C++17 + SFML 3** projetado como **ambiente de treinamento de IA via Machine Learning** — neuroevolução, Q-Learning, Policy Gradient, ou qualquer algoritmo que implemente a interface `reset() / step()`.

A motivação central não é o jogo em si, mas a qualidade da arquitetura como ambiente de RL: simulação determinística, desacoplada da renderização, executável sem janela (headless) e escalável a milhares de agentes simultâneos.

---

## Sumário

1. [Requisitos](#requisitos)
2. [Instalação e Build](#instalação-e-build)
3. [Modos de Execução](#modos-de-execução)
4. [Controles (modo janela)](#controles-modo-janela)
5. [Opções de linha de comando](#opções-de-linha-de-comando)
6. [Testes](#testes)
7. [Criando novos mapas](#criando-novos-mapas)
8. [Plugando um algoritmo de ML](#plugando-um-algoritmo-de-ml)
9. [Constantes configuráveis](#constantes-configuráveis)
10. [Estrutura de arquivos](#estrutura-de-arquivos)

---

## Requisitos

| Dependência | Versão mínima | Como instalar |
|---|---|---|
| macOS | 13+ (Ventura) | — |
| Xcode Command Line Tools | 15+ | `xcode-select --install` |
| CMake | 3.16+ | `brew install cmake` |
| SFML | **3.x** | `brew install sfml` |
| nlohmann/json | 3.11.3 | Baixado automaticamente pelo CMake |

> **Atenção:** o projeto usa a API do **SFML 3** (eventos com `std::optional`, enums com escopo, `sf::Text` com fonte no construtor). Não é compatível com SFML 2.

---

## Instalação e Build

```bash
# 1. Clone ou entre no diretório do projeto
cd racing-ml-sim

# 2. Instale as dependências (se ainda não instalou)
brew install cmake sfml

# 3. Configure e compile em Release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)

# O binário fica em:
./build/racing_sim
```

A build baixa `nlohmann/json` automaticamente via `FetchContent` na primeira execução do cmake — é necessário acesso à internet nessa etapa.

---

## Modos de Execução

### Modo janela (padrão)

Abre uma janela 900×700 com a pista e os carros. O **carro 0** (amarelo) é controlado pelo teclado. Os demais têm redes neurais com pesos aleatórios.

```bash
./build/racing_sim
```

Com múltiplos carros (útil para ver os NNs se movendo):

```bash
./build/racing_sim --population 10
```

### Modo headless

Roda a simulação sem abrir janela — ideal para treino de IA onde a renderização seria um gargalo.

```bash
./build/racing_sim --headless --population 1000
```

### Modo benchmark

Mede o tempo de wall-clock para simular 1 geração completa (todos os carros até `done`) e imprime o resultado. O target de performance é **< 5 segundos** para N=1000 carros × 60s × 60Hz em hardware Apple Silicon.

```bash
./build/racing_sim --benchmark --population 1000

# Exemplo de saída:
# Benchmark: population=1000 threads=8
#   Wall-clock: 0.48s  (~3600000 car-ticks total)
```

Comparar single vs multi-thread:

```bash
./build/racing_sim --benchmark --population 1000 --threads 1
./build/racing_sim --benchmark --population 1000 --threads 8
```

---

## Controles (modo janela)

| Tecla | Ação |
|---|---|
| `W` / `↑` | Acelerar |
| `S` / `↓` | Frear / ré |
| `A` / `←` | Virar à esquerda |
| `D` / `→` | Virar à direita |
| Fechar janela | Encerrar |

O carro 0 (amarelo) exibe os **raios de sensor** em debug: verde = distância longa, vermelho = perto da borda.

---

## Opções de linha de comando

```
--headless            Roda sem janela
--map <path>          Caminho do JSON do mapa (default: maps/map1.json)
--population <N>      Número de carros simultâneos (default: 1)
--seed <S>            Seed do RNG — garante reprodutibilidade (default: 42)
--benchmark           Mede throughput e sai
--threads <K>         Threads para atualização dos carros (default: hardware_concurrency)
--help / -h           Exibe ajuda
```

---

## Testes

```bash
# Via CTest (roda a partir da raiz do projeto)
ctest --test-dir build --output-on-failure

# Direto (mais verboso)
./build/racing_tests
```

Saída esperada:

```
=== Racing ML Sim Tests ===
Vec2: ok
Track geometry: ok
Sensor: ok
NeuralNetwork: ok
Determinism (single-thread): ok
Determinism (multi-thread): ok
GeneticAlgorithm: ok
Game headless episode: ok

Results: 79 passed, 0 failed
```

### O que os testes cobrem

| Grupo | O que verifica |
|---|---|
| Vec2 | length, normalized, rotated, dot, perpendicular |
| Track geometry | raycast com distância conhecida, bordas simétricas, `isInsideTrack`, erros de JSON |
| Sensor | leituras ∈ [0,1], raio central alcança borda |
| NeuralNetwork | forward pass com pesos conhecidos, round-trip save/load, rejeição de magic/version/topologia errados |
| Determinismo (single) | duas instâncias com mesma seed → trajetória idêntica bit-a-bit |
| Determinismo (multi) | 100 carros, 300 ticks, single vs multi-thread → fitness por carro idêntico |
| GeneticAlgorithm | initPopulation, setFitness, evolve incrementa geração |
| Headless episode | episódio termina e `episodeDone()` retorna `true` |

---

## Criando novos mapas

Mapas são arquivos JSON em `maps/`. Schema:

```json
{
  "name": "nome_do_mapa",
  "closed": true,
  "track_width": 120.0,
  "waypoints": [
    [x0, y0],
    [x1, y1],
    ...
  ],
  "obstacles": [
    { "type": "circle", "pos": [cx, cy], "radius": 30.0 },
    { "type": "rect",   "pos": [cx, cy], "size": [largura, altura] }
  ]
}
```

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `closed` | bool | sim | `true` = circuito fechado (volta), `false` = pista aberta (chegada no último waypoint) |
| `track_width` | float | sim | Largura total da pista em pixels. Bordas geradas com offset ±width/2 perpendicular à linha central |
| `waypoints` | array de [x,y] | sim | Mínimo 3 pontos. Define a **linha central** da pista em ordem |
| `obstacles` | array | não | Obstáculos estáticos: `circle` (pos, radius) ou `rect` (pos = centro, size) |

**Convenção de coordenadas:** origem no canto superior-esquerdo, x cresce para a direita, y cresce para baixo. Ângulos em radianos, sentido horário.

**Spawn:** todos os carros nascem no `waypoints[0]`, orientados na direção de `waypoints[1]`.

Carregar um mapa diferente:

```bash
./build/racing_sim --map maps/meu_mapa.json
```

---

## Plugando um algoritmo de ML

### Opção 1 — Single-agent RL (Q-Learning, PPO, SAC…)

Implemente `AIController` e use a interface `reset()`/`step()`:

```cpp
#include "AIController.h"
#include "Game.h"

class MeuAgente : public AIController {
public:
    Action decide(const Observation& obs) override {
        // obs[0..6]  → 7 leituras de raycast normalizadas [0,1]
        // obs[7]     → velocidade normalizada [0,1]
        // obs[8]     → ângulo para o próximo waypoint ∈ [-1,1]
        // obs[9]     → distância para o próximo waypoint ∈ [0,1]
        //
        // Retorne Action{throttle, steering} ambos ∈ [-1, 1]
        return Action{1.f, 0.f}; // exemplo: sempre acelera reto
    }
    void reset() override { /* limpa estado interno do agente */ }
};

int main() {
    SimConfig cfg;
    cfg.population = 1;

    Game game(cfg);
    game.setControllers({ std::make_unique<MeuAgente>() });

    Observation obs = game.reset();

    while (true) {
        // Seu agente escolhe a ação
        Action act = meuAgente.selectAction(obs);

        // Simulação avança 1 tick
        auto [next_obs, reward, done] = game.step(act);

        // Treino do agente com a transição (obs, act, reward, next_obs, done)
        meuAgente.train(obs, act, reward, next_obs, done);

        obs = next_obs;
        if (done) obs = game.reset();
    }
}
```

### Opção 2 — Neuroevolução com NeuralNetwork + GeneticAlgorithm

```cpp
#include "Game.h"
#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"

int main() {
    const int    POP  = 200;
    const int    GENS = 100;

    SimConfig cfg;
    cfg.population = POP;
    cfg.headless   = true;
    cfg.threads    = 0; // hardware_concurrency

    // Descobre quantos pesos a NN padrão tem
    NeuralNetwork refNet({OBS_SIZE, 8, 2}, 0);
    size_t wCount = refNet.getWeights().size();

    GeneticAlgorithm ga;
    ga.initPopulation(POP, wCount, /*seed=*/42);

    for (int gen = 0; gen < GENS; ++gen) {
        // Monta os controladores com os pesos da geração atual
        std::vector<std::unique_ptr<AIController>> ctrls;
        for (int i = 0; i < POP; ++i) {
            NeuralNetwork nn({OBS_SIZE, 8, 2}, (unsigned)i);
            nn.setWeights(ga.genomes()[i].weights);
            ctrls.push_back(std::make_unique<NeuralNetworkController>(std::move(nn)));
        }

        Game game(cfg);
        game.setControllers(std::move(ctrls));
        game.runHeadlessEpisode(); // roda a geração inteira

        // Coleta fitness e repassa ao GA
        auto fits = game.fitnesses();
        for (int i = 0; i < POP; ++i)
            ga.setFitness(i, fits[i]);

        std::cout << "Gen " << gen << "  best=" << *std::max_element(fits.begin(), fits.end()) << "\n";

        ga.evolve(); // seleção + crossover + mutação → nova geração
    }
}
```

### Opção 3 — Algoritmo completamente customizado

Basta implementar `AIController::decide`:

```cpp
class MeuNEAT : public AIController {
public:
    Action decide(const Observation& obs) override { /* ... */ }
};
```

E injetar via `game.setControllers(...)`. O `Game` não sabe nem liga para o que está dentro do controlador.

---

## Constantes configuráveis

Todas em `src/core/Constants.h`. Mudar qualquer uma exige recompilar.

| Constante | Valor padrão | Significado |
|---|---|---|
| `SIM_HZ` | 60 | Frequência da simulação (passos/segundo) |
| `DT` | 1/60 s | Timestep fixo de cada passo |
| `NUM_RAYS` | 7 | Número de raios do sensor (muda `OBS_SIZE` automaticamente) |
| `RAY_MAX_LEN` | 300 px | Distância máxima de cada raio; além disso retorna 1.0 |
| `MAX_SPEED` | 400 px/s | Velocidade máxima do carro |
| `ACCEL` | 300 px/s² | Aceleração com throttle positivo |
| `BRAKE` | 500 px/s² | Desaceleração/ré com throttle negativo |
| `DRAG` | 0.98 | Fator de atrito por tick (multiplica a velocidade) |
| `MAX_STEER` | 3.0 rad/s | Taxa máxima de giro a velocidade máxima |
| `EPISODE_TIMEOUT` | 60 s | Duração máxima de um episódio |
| `STALL_TIMEOUT` | 5 s | Tempo máximo sem progresso antes de `done` |
| `OBS_SIZE` | 10 | Tamanho fixo do vetor de observação (= `NUM_RAYS + 3`) |

Pesos do reward em `src/core/Types.h` (`RewardConfig`):

| Campo | Padrão | Significado |
|---|---|---|
| `w_progress` | 1.0 | Multiplicador do progresso por tick |
| `w_speed` | 0.1 | Bônus por andar rápido |
| `w_idle` | 0.05 | Penalidade por ficar parado |
| `w_finish` | 100.0 | Bônus por completar o circuito |
| `w_crash` | 50.0 | Penalidade por colisão |
| `idle_eps` | 5 px/s | Threshold de velocidade para penalidade de inatividade |

---

## Estrutura de arquivos

```
racing-ml-sim/
├── CMakeLists.txt          # Build: SFML 3 + nlohmann/json (FetchContent)
├── README.md               # Este arquivo
├── ARCHITECTURE.md         # Documentação técnica detalhada
├── assets/
│   └── DejaVuSans.ttf      # Fonte open-source para o HUD
├── maps/
│   └── map1.json           # Circuito de exemplo (8 waypoints, 2 obstáculos)
├── src/
│   ├── core/
│   │   ├── Vec2.h          # Vetor 2D matemático, header-only, sem SFML
│   │   ├── Constants.h     # Todas as constantes da simulação
│   │   └── Types.h         # Observation, Action, StepResult, RewardConfig, SimConfig
│   ├── AIController.h      # Interface abstrata: decide(Observation) → Action
│   ├── Track.h / .cpp      # Pista: JSON, bordas, raycast, progresso
│   ├── Sensor.h / .cpp     # 7 raios normalizados em leque de 180°
│   ├── Car.h / .cpp        # Física, observação, reward, condições de done
│   ├── NeuralNetwork.h/.cpp# MLP feedforward + serialização binária + NNController
│   ├── GeneticAlgorithm.h/.cpp # GA: init, evolve, crossover, mutação
│   ├── Game.h / .cpp       # reset/step (RL), tick batch, thread pool
│   ├── Renderer.h / .cpp   # ÚNICA camada SFML: pista, carros, raios, HUD
│   ├── HumanController.h/.cpp  # Leitura de teclado → Action (depende de SFML)
│   └── main.cpp            # Parse de args, dispatch para os modos
└── tests/
    └── test_main.cpp       # 79 testes: Vec2, geometria, NN, determinismo, GA
```
