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

### Treinamento headless (mais rápido — paralelo)

Loop geracional sem janela. Todos os **M mapas × P carros** de uma geração são avaliados em paralelo num pool de threads — sem recarregar mapas entre gerações.

```bash
# 1000 carros, 5000 gerações (recomendado para treino sério)
./build/racing_sim --train --headless --population 1000 --generations 5000

# Controlar número de threads (default: todos os cores disponíveis)
./build/racing_sim --train --headless --population 1000 --generations 5000 --threads 8

# Controlar seed e pasta de saída
./build/racing_sim --train --headless --seed 123 --out resultados/

# Retomar a partir de um campeão salvo
./build/racing_sim --train --headless --load out/best.rnnw --generations 5000
```

Saída no terminal (uma linha por geração):

```
gen    1/5000  agg= -0.308 | m0= 0.333 m1= 0.151 m2= 0.067 m3= 0.037 m4= 0.082 m5= 0.021 |  mean= -0.798  std= 0.242  done=0/1000  [col=480 stall=520 timeout=0]
gen    2/5000  agg=  0.412 | m0= 0.891 m1= 0.694 m2= 0.138 m3= 0.412 m4= 0.056 m5= 0.206 |  mean= -0.241  std= 0.381  done=8/1000  [col=320 stall=672 timeout=0]
...
```

Colunas: `agg` = fitness agregado (min entre mapas), `m0..m5` = melhor fitness normalizado por mapa, `done` = carros que completaram o circuito.

### Treinamento multi-mapa (generalização)

Treina em vários mapas simultaneamente para produzir um agente que generaliza, em vez de memorizar um único circuito.

```bash
# Automático: usa todos os *.json em maps/, split 6 treino / 2 val
./build/racing_sim --train --headless --population 200 --generations 150

# Explicitar os mapas de treino e validação
./build/racing_sim --train --headless \
  --train-maps maps/map1.json,maps/map2.json,maps/map3.json \
  --val-maps   maps/map4.json,maps/map5.json \
  --generations 100

# Usar média em vez de mínimo para o fitness agregado
./build/racing_sim --train --headless --fitness-agg mean --generations 100
```

O **fitness agregado** (coluna `agg=` no terminal) é por padrão o **mínimo** entre mapas — força o agente a não ignorar nenhuma pista. Com `--fitness-agg mean` usa a média, mais tolerante a mapas difíceis.

Arquivos gerados em `out/` (ou `--out <dir>`):

| Arquivo | Conteúdo |
|---|---|
| `best.rnnw` | Pesos do **campeão global** (sobrescrito quando há novo melhor) |
| `gen_0001.rnnw` … `gen_NNNN.rnnw` | Snapshot do melhor da geração N |

### Treinamento com janela (serial — visualização)

Igual ao headless, mas abre a janela para assistir à evolução ao vivo. **Atenção:** o modo janela é serial — os mapas são percorridos um por um em sincronia com o render. Para treinos grandes use `--headless`.

```bash
./build/racing_sim --train --population 100 --generations 200
```

Com múltiplos mapas, a janela alterna automaticamente entre os mapas de treino ao longo de cada geração — o nome do mapa atual aparece na tela. Pressione **`T`** para alternar entre **tempo real** (60 Hz) e **turbo** (máxima velocidade).

### Assistir uma rede treinada

Carrega um arquivo `.rnnw` e abre a janela com 1 carro dirigindo em loop.

```bash
./build/racing_sim --watch out/best.rnnw
# ou equivalente:
./build/racing_sim --load  out/best.rnnw
```

### Human vs AI

Você controla o carro amarelo; a rede salva controla o verde. Use `←→↑↓` ou `WASD`.

```bash
./build/racing_sim --versus out/best.rnnw

# Em um mapa específico:
./build/racing_sim --versus out/best.rnnw --map maps/map3.json
```

A corrida reinicia automaticamente quando os dois terminam. Pressione **Restart** (ou `R`) para recomeçar a qualquer momento.

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
| `W` / `↑` | default, `--versus` | Acelerar |
| `S` / `↓` | default, `--versus` | Frear |
| `A` / `←` | default, `--versus` | Virar à esquerda |
| `D` / `→` | default, `--versus` | Virar à direita |
| `T` | `--train` | Alternar tempo real ↔ turbo |
| `< Mapa` / `Mapa >` | todos (janela) | Trocar de pista |
| Restart | todos (janela) | Reiniciar episódio |
| Fechar janela | todos | Encerrar |

O carro 0 (amarelo) exibe os **raios de sensor**: verde = distância longa, vermelho = perto da borda. Em `--versus`, amarelo = você, verde = IA.

---

## Opções de linha de comando

```
Básico
  --headless              Roda sem janela
  --map <path>            Caminho do JSON do mapa (default: maps/map1.json)
  --population <N>        Número de carros simultâneos (default: 1)
  --seed <S>              Seed do RNG — garante reprodutibilidade (default: 42)
  --threads <K>           Threads para atualização dos carros (default: hardware_concurrency)
  --benchmark             Mede throughput e sai

Treinamento
  --train                 Ativa o loop geracional de treinamento
  --algo <nome>           Algoritmo: genetic | random_search | hillclimb (default: genetic)
  --generations <N>       Número de gerações (default: 100)
  --out <dir>             Diretório de saída dos pesos (default: out/)
  --load <arquivo.rnnw>   Com --train: semeia população do campeão. Sem --train: abre modo watch

Generalização multi-mapa
  --train-maps <a,b,...>  Lista de mapas de treino separados por vírgula
  --val-maps <x,y>        Mapas de validação (não usados na seleção, apenas métricas)
  --fitness-agg <modo>    Como agregar fitness entre mapas: min (default) | mean
  (sem --train-maps: usa todos os *.json em maps/, split 6 treino / 2 val automático)

Watch / Interativo
  --watch <arquivo.rnnw>  Abre janela com a rede salva dirigindo (sem treino)
  --versus <arquivo.rnnw> Human (amarelo, WASD/setas) vs IA (verde) com rede salva

  --help / -h             Exibe ajuda
```

**Precedência de modo:** `--benchmark` > `--watch` > `--versus` > `--train` > `--load` (sem train = watch) > modo janela padrão.

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

Results: 286 passed, 0 failed
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
| Reverse blocked | `MAX_REVERSE_SPEED=0` impede velocidade negativa |
| Reverse penalty | `w_reverse > 0` acumula penalidade ao andar de ré |
| Regress penalty | `w_regress > 0` acumula penalidade ao regredir no percurso |
| Stall no-progress | carro sem avanço por `STALL_TIMEOUT` termina com `DoneReason::Stall` |

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
        // obs[0..12] → 13 leituras de raycast normalizadas [0,1]
        // obs[13]    → velocidade normalizada [0,1]
        // obs[14]    → ângulo para o próximo waypoint ∈ [-1,1]
        // obs[15]    → distância para o próximo waypoint ∈ [0,1]
        // obs[16]    → ângulo para o waypoint seguinte (lookahead 2) ∈ [-1,1]
        // obs[17..26]→ 5x (curvatura, speed_excess) para lookaheads 1-5
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
| `NUM_RAYS` | 13 | Número de raios do sensor (muda `OBS_SIZE` automaticamente) |
| `RAY_MAX_LEN` | 400 px | Distância máxima de cada raio; além disso retorna 1.0 |
| `MAX_SPEED` | 400 px/s | Velocidade máxima do carro |
| `MAX_REVERSE_SPEED` | 0 px/s | Velocidade máxima de ré (0 = ré desativada) |
| `ACCEL` | 300 px/s² | Aceleração com throttle positivo |
| `BRAKE` | 500 px/s² | Desaceleração com throttle negativo |
| `DRAG` | 0.98 | Fator de atrito por tick (multiplica a velocidade) |
| `MAX_STEER` | 3.0 rad/s | Taxa máxima de giro |
| `MAX_LAT_ACCEL` | 650 px/s² | Limite de grip (yawRate ≤ MAX_LAT_ACCEL/v) |
| `EPISODE_TIMEOUT` | 60 s | Duração máxima de um episódio |
| `STALL_TIMEOUT` | 2 s | Tempo sem progresso (ou parado) antes de `done` |
| `STALL_SPEED` | 4 px/s | Threshold de velocidade mínima para o stall timer |
| `STALL_PROGRESS_MIN` | 0.05 | Avanço mínimo de waypoint para resetar o timer de stall |
| `OBS_SIZE` | 27 | Tamanho fixo do vetor de observação (= `NUM_RAYS + 14`) |
| `NN_HIDDEN` | 32 | Neurônios na camada oculta da rede padrão |

**Vetor de observação (27 floats):**

| Índices | Conteúdo |
|---|---|
| `[0..12]` | 13 leituras de raycast normalizadas [0,1] |
| `[13]` | Velocidade normalizada [0,1] |
| `[14]` | Ângulo para o próximo waypoint ∈ [-1,1] |
| `[15]` | Distância para o próximo waypoint ∈ [0,1] |
| `[16]` | Ângulo para o waypoint seguinte (lookahead 2) ∈ [-1,1] |
| `[17..26]` | 5 × (curvatura assinada, speed_excess) para lookaheads 1–5 |

Pesos do reward em `src/core/Types.h` (`RewardConfig`):

| Campo | Padrão | Significado |
|---|---|---|
| `w_progress` | 1.0 | Multiplicador do progresso máximo acumulado |
| `w_speed` | 0.1 | Bônus de velocidade em trechos retos |
| `w_checkpoint` | 0.5 | Bônus por passar waypoints curvos |
| `w_finish` | 200.0 | Bônus por completar o circuito |
| `w_time` | 1.0 | Bônus de tempo restante ao completar |
| `w_crash` | 50.0 | Penalidade por colisão |
| `w_reverse` | 0.5 | Penalidade por andar de ré |
| `w_regress` | 2.0 | Penalidade por regredir no percurso |
| `w_curve` | 0.0 | Penalidade por alta velocidade em curvas (desativado) |

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
│   ├── Sensor.h / .cpp     # 13 raios normalizados em leque de 180°
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
    └── test_main.cpp       # 286 testes: Vec2, geometria, NN, determinismo, GA, Trainers, TrainingSession, reward
```
