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

Abre uma janela 900×700. O **carro 0** (amarelo) é controlado pelo teclado; os demais têm redes neurais com pesos aleatórios.

```bash
./build/racing_sim
./build/racing_sim --population 10   # mais carros com NN aleatória
```

### Treinamento headless (mais rápido)

Loop geracional sem janela: roda N gerações, imprime stats e salva os pesos.

```bash
# 100 gerações, 200 carros, algoritmo genético (default)
./build/racing_sim --train --headless --population 200 --generations 100

# Trocar o algoritmo
./build/racing_sim --train --headless --algo random_search --population 100 --generations 50
./build/racing_sim --train --headless --algo hillclimb     --population 100 --generations 50

# Controlar seed e saída
./build/racing_sim --train --headless --seed 123 --out resultados/

# Retomar a partir de um campeão salvo
./build/racing_sim --train --headless --load out/best.rnnw --generations 50
```

Saída no terminal (uma linha por geração):

```
gen    1/100  best=  344.69  mean=  -32.75  std= 62.26  done=0/200
gen    2/100  best=  412.03  mean=   -8.11  std= 71.40  done=0/200
...
```

Arquivos gerados em `out/` (ou `--out <dir>`):

| Arquivo | Conteúdo |
|---|---|
| `best.rnnw` | Pesos do **campeão global** (sobrescrito quando há novo melhor) |
| `gen_0001.rnnw` … `gen_NNNN.rnnw` | Snapshot do melhor da geração N |

### Treinamento com janela

Igual ao headless, mas abre a janela para assistir à evolução ao vivo. Mostra HUD geracional e gráfico do melhor fitness por geração.

```bash
./build/racing_sim --train --population 100 --generations 50
```

Pressione **`T`** para alternar entre **tempo real** (assiste a corrida a 60 Hz) e **turbo** (avança várias gerações por frame, mostrando só o estado atual + gráfico).

### Assistir uma rede treinada

Carrega um arquivo `.rnnw` e abre a janela com 1 carro dirigindo em loop.

```bash
./build/racing_sim --watch out/best.rnnw
# ou equivalente:
./build/racing_sim --load  out/best.rnnw
```

### Modo headless sem treino

Roda 1 episódio completo e imprime o tempo.

```bash
./build/racing_sim --headless --population 1000
```

### Modo benchmark

Mede throughput de simulação e sai.

```bash
./build/racing_sim --benchmark --population 1000

# Exemplo de saída:
# Benchmark: population=1000 threads=8
#   Wall-clock: 0.48s  (~3600000 car-ticks total)
```

```bash
# Comparar single vs multi-thread
./build/racing_sim --benchmark --population 1000 --threads 1
./build/racing_sim --benchmark --population 1000 --threads 8
```

---

## Controles (modo janela)

| Tecla | Modo | Ação |
|---|---|---|
| `W` / `↑` | default | Acelerar |
| `S` / `↓` | default | Frear / ré |
| `A` / `←` | default | Virar à esquerda |
| `D` / `→` | default | Virar à direita |
| `T` | `--train` | Alternar tempo real ↔ turbo |
| Fechar janela | todos | Encerrar |

O carro 0 (amarelo) exibe os **raios de sensor**: verde = distância longa, vermelho = perto da borda.

---

## Opções de linha de comando

```
--headless              Roda sem janela
--map <path>            Caminho do JSON do mapa (default: maps/map1.json)
--population <N>        Número de carros simultâneos (default: 1)
--seed <S>              Seed do RNG — garante reprodutibilidade (default: 42)
--threads <K>           Threads para atualização dos carros (default: hardware_concurrency)
--benchmark             Mede throughput e sai

--train                 Ativa o loop geracional de treinamento
--algo <nome>           Algoritmo: genetic | random_search | hillclimb (default: genetic)
--generations <N>       Número de gerações (default: 100)
--out <dir>             Diretório de saída dos pesos (default: out/)
--load <arquivo.rnnw>   Com --train: semeia população do campeão. Sem --train: abre modo watch
--watch <arquivo.rnnw>  Abre janela com a rede salva dirigindo (sem treino)

--help / -h             Exibe ajuda
```

**Precedência de modo:** `--benchmark` > `--watch` > `--train` > `--load` (sem train = watch) > modo janela padrão.

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
Trainers weight count: ok
Trainers determinism: ok
Trainers elitism: ok
Trainers resume: ok
TrainingSession headless: ok
Game headless episode: ok

Results: 136 passed, 0 failed
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
| Trainers weight count | `weights(i).size() == weightCount` para todos os algoritmos e indivíduos |
| Trainers determinism | dois trainers com mesma seed + mesma sequência de fitness → população idêntica após evolve |
| Trainers elitism | `random_search` e `hillclimb` mantêm best fitness global não-decrescente |
| Trainers resume | `initFromWeights` preserva o campeão em `weights(0)` |
| TrainingSession headless | run de 3 gerações gera `best.rnnw` + `gen_000N.rnnw` recarregáveis |
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

### Opção 2 — Neuroevolução via CLI (recomendado)

Use diretamente a linha de comando — não precisa escrever código:

```bash
# Treino headless, 100 gerações, 200 carros, GA
./build/racing_sim --train --headless --population 200 --generations 100 --out out/

# Assistir o campeão depois
./build/racing_sim --watch out/best.rnnw

# Continuar o treino a partir do campeão
./build/racing_sim --train --headless --load out/best.rnnw --generations 50
```

### Opção 3 — Neuroevolução programática (TrainingSession)

Para integrar o loop de treino no seu próprio código:

```cpp
#include "Trainers.h"
#include "Training.h"

int main() {
    SimConfig cfg;
    cfg.population = 200;
    cfg.headless   = true;

    TrainingSession session(cfg, makeTrainer("genetic"), /*generations=*/100, "out/");
    session.runAll(); // imprime stats por geração e salva best.rnnw + gen_NNNN.rnnw

    // Inspecionar resultado
    const auto& stats = session.lastStats();
    std::cout << "Melhor fitness final: " << stats.bestFitness << "\n";
}
```

Ou controlar tick a tick (para interlaçar com render):

```cpp
session.beginGeneration();
while (!session.generationComplete()) session.tick();
session.endGeneration(); // coleta fitness, salva, evolve
```

### Opção 4 — Neuroevolução manual (GeneticAlgorithm direto)

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

### Opção 5 — Algoritmo completamente customizado

Implemente a interface `Trainer` para usar seu próprio algoritmo com o loop geracional existente:

```cpp
#include "Trainer.h"

class MeuCMAES : public Trainer {
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

// Usar com TrainingSession
TrainingSession session(cfg, std::make_unique<MeuCMAES>(), 100, "out/");
session.runAll();
```

Ou, para algo ainda mais simples, basta implementar `AIController::decide` e injetar via `game.setControllers(...)`. O `Game` não sabe nem liga para o que está dentro do controlador.

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
│   ├── NeuralNetwork.h/.cpp# MLP feedforward + serialização binária RNNW + NNController
│   ├── GeneticAlgorithm.h/.cpp # GA: init, seedFrom, evolve, crossover, mutação
│   ├── Trainer.h           # Interface Trainer + struct GenerationStats (sem SFML)
│   ├── Trainers.h / .cpp   # GeneticTrainer, RandomSearchTrainer, HillClimbTrainer + makeTrainer()
│   ├── Training.h / .cpp   # TrainingSession: loop geracional, stats, save/load (sem SFML)
│   ├── Game.h / .cpp       # reset/step (RL), tick batch, thread pool
│   ├── Renderer.h / .cpp   # ÚNICA camada SFML: pista, carros, HUD, gráfico fitness
│   ├── HumanController.h/.cpp  # Leitura de teclado → Action (depende de SFML)
│   └── main.cpp            # Parse de args, dispatch para todos os modos
└── tests/
    └── test_main.cpp       # 136 testes: Vec2, geometria, NN, determinismo, GA, Trainers, TrainingSession
```
