# Racing ML Sim

Jogo 2D de corrida top-down escrito em **C++17 + SFML 3** projetado como **ambiente de treinamento de IA via Machine Learning** — neuroevolução, Q-Learning, Policy Gradient, ou qualquer algoritmo que implemente a interface `reset() / step()`.

A motivação central não é o jogo em si, mas a qualidade da arquitetura como ambiente de RL: simulação determinística, desacoplada da renderização, executável sem janela (headless) e escalável a milhares de agentes simultâneos.

---

## Sumário

1. [Requisitos](#requisitos)
2. [Instalação e Build](#instalação-e-build)
3. [Arquivo de configuração (train.json)](#arquivo-de-configuração-trainjson)
4. [Modos de Execução](#modos-de-execução)
5. [Controles (modo janela)](#controles-modo-janela)
6. [Opções de linha de comando](#opções-de-linha-de-comando)
7. [Testes](#testes)
8. [Criando novos mapas](#criando-novos-mapas)
9. [Plugando um algoritmo de ML](#plugando-um-algoritmo-de-ml)
10. [Constantes configuráveis](#constantes-configuráveis)
11. [Estrutura de arquivos](#estrutura-de-arquivos)


---

## Requisitos

| Dependência | Versão mínima |
|---|---|
| Compilador C++17 | clang 14+ / gcc 11+ / MSVC 2022 |
| CMake | 3.16+ |
| SFML | **3.x** |
| nlohmann/json | 3.11.3 (baixado automaticamente pelo CMake) |

> **Atenção:** o projeto usa a API do **SFML 3** (eventos com `std::optional`, enums com escopo, `sf::Text` com fonte no construtor). Não é compatível com SFML 2.

O projeto roda em **macOS, Linux e Windows** — não há nenhuma dependência de plataforma no código.

---

## Instalação e Build

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

> Em distros mais antigas (ex: Ubuntu 22.04), o `libsfml-dev` pode instalar o SFML 2. Verifique com `apt show libsfml-dev | grep Version`. Se for 2.x, [compile o SFML 3 do fonte](https://github.com/SFML/SFML) ou use uma distro mais recente.

### Windows

**Visual Studio 2022 + vcpkg**
```powershell
# Instale o Visual Studio 2022 com o workload "Desenvolvimento para Desktop com C++"
# Instale vcpkg: https://github.com/microsoft/vcpkg
vcpkg install sfml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
# Binário: build\Release\racing_sim.exe
```

**MSYS2 / MinGW-w64**
```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-sfml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
cmake --build build -j$(nproc)
./build/racing_sim.exe
```

---

A build baixa `nlohmann/json` automaticamente via `FetchContent` na primeira execução do cmake — é necessário acesso à internet nessa etapa.

---

## Arquivo de configuração (train.json)

`train.json` é o arquivo central de configuração do experimento. Parametrize-o corretamente antes de treinar — o simulador detecta e carrega o arquivo automaticamente ao rodar.

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

As chaves do JSON usam underscore e espelham os flags CLI: `train_maps`, `cvar_alpha`, `random_spawn`, `episodes_per_eval`, etc. Arrays são aceitos em `train_maps`, `val_maps`, `test_maps`, `augment` e `map_weights`.

Para usar uma config alternativa (ex: experimentos paralelos):

```bash
./build/racing_sim --train --config experimento_b.json
```

---

## Modos de Execução

### Treinamento

Com o `train.json` configurado, rode:

```bash
./build/racing_sim --train
```

Com `headless: true` no `train.json`, o treino roda sem janela — muito mais rápido para populações grandes, pois todos os **M mapas × P carros** são avaliados em paralelo num pool de threads.

Saída no terminal (uma linha por geração):

```
gen    1/1500  agg= -0.308 | m0= 0.333 m1= 0.151 m2= 0.067 |  mean= -0.798  std= 0.242  done=0/1000  [col=480 stall=520 timeout=0]
gen    2/1500  agg=  0.412 | m0= 0.891 m1= 0.694 m2= 0.138 |  mean= -0.241  std= 0.381  done=8/1000  [col=320 stall=672 timeout=0]
...
```

Colunas: `agg` = fitness agregado (modo configurável via `--fitness-agg`), `m0..mN` = melhor fitness normalizado por mapa, `done` = carros que completaram o circuito.

Arquivos gerados no diretório definido em `out`:

| Arquivo | Conteúdo |
|---|---|
| `best.rnnw` | Pesos do **campeão global** (sobrescrito quando há novo melhor) |
| `gen_0001.rnnw` … `gen_NNNN.rnnw` | Snapshot do melhor da geração N |
| `training_log.csv` | Métricas de treino por geração (requer `--log-csv`) |
| `held_out_log.csv` | Métricas de validação por geração (requer `--log-csv`) |
| `test_log.csv` | Métricas do conjunto de teste (requer `--log-csv` + `test_maps`) |

### Treinamento com janela (visualização)

Quando `headless: false` no `train.json` (ou sem `--headless`), a janela abre para acompanhar a evolução ao vivo. **Atenção:** o modo janela é serial — os mapas são percorridos um por um em sincronia com o render.

```bash
./build/racing_sim --train
```

Com múltiplos mapas, a janela alterna automaticamente entre os mapas de treino. Pressione **`T`** para alternar entre **tempo real** (60 Hz) e **turbo** (máxima velocidade).

### Combatendo overfitting (diversidade, anti-memorização e seleção por held-out)

Quando o agente decora os mapas de treino mas falha em pistas novas, há quatro alavancas configuráveis via `train.json` ou CLI:

```bash
./build/racing_sim --train --headless --population 1000 --generations 1000 \
  --train-maps maps/map1.json,maps/map4.json,maps/map5.json \
  --augment mirror,reverse,width:0.85,width:1.15 \   # B) variantes estáticas dos mapas base
  --procedural-train 8 \                              # A) pistas aleatórias geradas (semeadas)
  --random-spawn --episodes-per-eval 3 --sensor-noise 0.02 \  # C) anti-memorização por episódio
  --val-maps maps/map7.json --test-maps maps/map8.json \
  --select-by-val --val-select-topk 10                # D) best.rnnw escolhido pela validação
```

- **A — diversidade:** `--procedural-train K` / `--procedural-val K` geram K pistas aleatórias (determinísticas pela seed); `--proc-width-min/--proc-width-max` controlam a faixa de largura.
- **B — augmentation:** `--augment mirror,reverse,width:<f>` adiciona variantes de cada mapa base ao treino.
- **C — anti-memorização:** `--random-spawn` (início aleatório na pista; progresso medido relativo ao spawn), `--episodes-per-eval N` (+ `--episode-agg mean|min`), `--sensor-noise <σ>`.
- **D — seleção por held-out:** `--select-by-val` salva `best.rnnw` pelo **melhor desempenho na validação** (entre os top-K do treino), em vez do fitness de treino. `--test-maps` é **só relatório** (nunca usado na seleção).

Acompanhe a curva de validação/teste ao vivo com `python3 tools/watch_training.py <out_dir>`.

### Modo janela (explorar / jogar)

Abre uma janela 900×700. O **carro 0** (amarelo) é controlado pelo teclado; os demais têm redes neurais com pesos aleatórios.

```bash
./build/racing_sim
./build/racing_sim --population 10   # mais carros com NN aleatória
```

### Assistir uma rede treinada

Carrega um arquivo `.rnnw` e abre a janela com 1 carro dirigindo em loop.

```bash
./build/racing_sim --watch out_v2/best.rnnw
# ou equivalente:
./build/racing_sim --load out_v2/best.rnnw
```

### Human vs AI

Você controla o carro amarelo; a rede salva controla o(s) verde(s). Use `←→↑↓` ou `WASD`.

```bash
./build/racing_sim --versus out_v2/best.rnnw

# Múltiplos AIs (--population N), com ruído nos pesos para divergirem entre si:
./build/racing_sim --versus out_v2/best.rnnw --population 5
./build/racing_sim --versus out_v2/best.rnnw --population 5 --versus-noise 0.05

# Em um mapa específico:
./build/racing_sim --versus out_v2/best.rnnw --map maps/map3.json
```

A corrida reinicia automaticamente quando todos terminam. Pressione **Restart** (ou `R`) para recomeçar a qualquer momento.

### Modo headless sem treino

Roda 1 episódio completo e imprime o tempo.

```bash
./build/racing_sim --headless --population 1000
```

### Modo benchmark

Mede throughput de simulação e sai.

```bash
./build/racing_sim --benchmark --population 1000

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

Referência completa de todos os flags disponíveis. O uso normal é configurar o experimento em `train.json` — os flags abaixo são para overrides pontuais e modos especiais.

```
Config file
  --config <arquivo.json> Carrega defaults de um arquivo JSON (flags CLI sobrescrevem)
  (auto-detect)           train.json na pasta atual é carregado automaticamente se existir

Básico
  --headless              Roda sem janela
  --map <path>            Caminho do JSON do mapa (default: maps/map1_chicanes_infernais.json)
  --population <N>        Número de carros simultâneos (default: 1)
  --seed <S>              Seed do RNG — mesma seed → mesmo treino (default: 42)
  --threads <K>           Threads para atualização dos carros (default: hardware_concurrency)
  --benchmark             Mede throughput e sai

Treinamento
  --train                 Ativa o loop geracional de treinamento
  --algo <nome>           Algoritmo: genetic | random_search | hillclimb (default: genetic)
  --generations <N>       Número de gerações (default: 100)
  --out <dir>             Diretório de saída dos pesos (default: out/)
  --load <arquivo.rnnw>   Com --train: semeia população do campeão. Sem --train: abre modo watch
  --log-csv               Salva training_log.csv, held_out_log.csv (e test_log.csv se --test-maps) em <out>
  --hidden <N>            Neurônios na camada oculta (default: 32; exige recompilar se mudar Constants.h)
  --episode-timeout <s>   Duração máxima de um episódio em segundos (default: 30)

Generalização multi-mapa
  --train-maps <a,b,...>  Lista de mapas de treino separados por vírgula
  --val-maps <x,y>        Mapas de validação (seleção quando --select-by-val)
  --test-maps <x,y>       Mapas de teste: só relatório (gera test_log.csv), nunca usados na seleção
  --fitness-agg <modo>    Agrega fitness entre mapas: cvar-rank (default) | min | mean | cvar-raw
  --cvar-alpha <α>        Fração de cauda CVaR ∈ (0,1] (default: 0.5). Usado com cvar-rank/cvar-raw
  --map-norm <modo>       Normalização por mapa antes do CVaR: zscore (default) | minmax | progress
                          Ignorado em cvar-rank (ranks são invariantes à escala)
  --map-weights <w,...>   Pesos por mapa para cvar-rank (vírgula, um por mapa de treino; default: todos 1.0)
  --progressive-frac <f>  Fração da população avaliada em mapas além do primeiro (default: 1.0)
  --finetune-map <path>   Substitui map[0] por este JSON — útil para refinar em nova pista sem retreinar tudo

Curriculum (desligado por padrão)
  --curriculum <modo>     linear (default) | none | explicit
  --curriculum-start <N>  Geração em que o segundo mapa entra (linear, default: 2)
  --curriculum-step <N>   Gerações entre adição de cada mapa subsequente (linear, default: 15)
  --curriculum-schedule <g,...>  Thresholds de geração para modo explicit (M−1 valores)
  --curriculum-pin <i,...>       Índices de mapas sempre ativos (independente do curriculum)

Generalização (combate a overfitting — todas desligadas por padrão)
  --augment <lista>       Variantes de treino: mirror,reverse,width:0.85,width:1.15
  --procedural-train <K>  Gera K pistas aleatórias (semeadas) no conjunto de treino
  --procedural-val <K>    Gera K pistas aleatórias (semeadas) no conjunto de validação
  --proc-width-min <w>    Largura mínima das pistas procedurais (default 55)
  --proc-width-max <w>    Largura máxima das pistas procedurais (default 110)
  --dump-gen-maps <dir>   Salva pistas augmentadas+procedurais como JSON em <dir> antes de treinar
  --random-spawn          Início aleatório na pista a cada episódio de treino
  --sensor-noise <s>      Ruído gaussiano (desvio) nas leituras de raio no treino
  --episodes-per-eval <N> Episódios por (genoma,mapa), agregados (default 1)
  --episode-agg <modo>    Combina episódios: mean (default) | min
  --select-by-val         Salva best.rnnw pela validação (top-K) em vez do fitness de treino
  --val-select-topk <T>   Genomas de treino avaliados na validação para seleção (default 1)

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
| `DRAG` | 0.98 | Fator de atrito por tick (0.98^60 ≈ 0.30 em 1s) |
| `MAX_STEER` | 3.0 rad/s | Taxa máxima de giro |
| `MAX_LAT_ACCEL` | 650 px/s² | Limite de grip lateral (yawRate ≤ MAX_LAT_ACCEL/v) |
| `EPISODE_TIMEOUT` | 30 s | Duração máxima de um episódio (configurável via `--episode-timeout`) |
| `STALL_TIMEOUT` | 2 s | Tempo sem progresso mínimo antes de `done` |
| `STALL_SPEED` | 4 px/s | Velocidade abaixo da qual o timer de stall corre |
| `STALL_PROGRESS_MIN` | 0.003 | Avanço mínimo de arco (fração do lap) para resetar o stall timer |
| `NUM_LOOKAHEADS` | 5 | Quantos pontos de lookahead de curvatura o carro enxerga |
| `OBS_SIZE` | 26 | Tamanho do vetor de observação (= `NUM_RAYS + 3 + 2 × NUM_LOOKAHEADS`) |
| `NN_HIDDEN` | 32 | Neurônios na camada oculta (sobrescrito por `--hidden`) |

**Vetor de observação (26 floats):**

| Índices | Conteúdo |
|---|---|
| `[0..12]` | 13 leituras de raycast normalizadas [0,1] — leque de 180° à frente |
| `[13]` | Velocidade normalizada `\|v\| / MAX_SPEED` ∈ [0,1] |
| `[14]` | Offset lateral em relação à linha central ∈ [-1,1] (positivo = à direita) |
| `[15]` | Heading error (ângulo do carro vs tangente da pista) ∈ [-1,1] |
| `[16,17]` | Lookahead 1 — (curvatura assinada ∈ [-1,1], speed_excess ∈ [-1,1]) |
| `[18,19]` | Lookahead 2 |
| `[20,21]` | Lookahead 3 |
| `[22,23]` | Lookahead 4 |
| `[24,25]` | Lookahead 5 (mais distante) |

`speed_excess` = quão rápido o carro vai em relação à velocidade segura para aquela curva (positivo = rápido demais).

Pesos do reward em `src/core/Types.h` (`RewardConfig`):

| Campo | Padrão | Significado |
|---|---|---|
| `w_progress` | 200.0 | Multiplicador do progresso máximo de arco ∈ [0,1] |
| `w_speed` | 0.3 | Bônus por velocidade enquanto avança (incentiva não parar) |
| `w_checkpoint` | 5.0 | Bônus por waypoint de design cruzado, escalado pela curvatura local |
| `w_finish` | 100.0 | Bônus por completar o circuito |
| `w_time` | 2.0 | Bônus de tempo restante ao completar (`w_time × (timeout − t)`) |
| `w_crash` | 15.0 | Penalidade por colisão |
| `w_reverse` | 0.5 | Penalidade acumulada por andar de ré (por unidade de velocidade negativa/s) |
| `w_regress` | 2.0 | Penalidade por regredir no percurso (maxProgress − currentProgress)/s |
| `w_curve` | 0.0 | Penalidade por alta velocidade em curvas (desativado por padrão) |

---

## Estrutura de arquivos

```
racing-ml-sim/
├── CMakeLists.txt          # Build: SFML 3 + nlohmann/json (FetchContent)
├── README.md               # Este arquivo
├── README-en.md            # Versão em inglês
├── ARCHITECTURE.md         # Documentação técnica detalhada
├── train.json              # Config padrão do experimento (carregado automaticamente)
├── assets/
│   └── DejaVuSans.ttf      # Fonte open-source para o HUD
├── maps/
│   ├── map1_chicanes_infernais.json  # Pista com sequência de chicanes
│   ├── map4_obstaculos.json          # Pista com obstáculos estáticos
│   ├── map5_tecnico_avancado.json    # Pista técnica avançada
│   ├── map7_pesadelo.json            # Mapa de validação
│   └── map8_caos_total.json          # Mapa de teste (held-out)
├── src/
│   ├── core/
│   │   ├── Vec2.h          # Vetor 2D matemático, header-only, sem SFML
│   │   ├── Constants.h     # Todas as constantes da simulação
│   │   ├── Types.h         # Observation, Action, StepResult, RewardConfig, SimConfig
│   │   └── TrackGen.h/.cpp # Geração procedural de pistas e augmentation (mirrorX, reverse, scaleWidth)
│   ├── AIController.h      # Interface abstrata: decide(Observation) → Action
│   ├── Track.h / .cpp      # Pista: JSON/in-memory, bordas Catmull-Rom, raycast, progresso por arco
│   ├── Sensor.h / .cpp     # 13 raios normalizados em leque de 180°
│   ├── Car.h / .cpp        # Física, observação (26 floats), reward, condições de done
│   ├── NeuralNetwork.h/.cpp# MLP feedforward + serialização binária RNNW + NNController
│   ├── GeneticAlgorithm.h/.cpp # GA: init, seedFrom, evolve, crossover, mutação
│   ├── Trainer.h           # Interface Trainer + struct GenerationStats (sem SFML)
│   ├── Trainers.h / .cpp   # GeneticTrainer, RandomSearchTrainer, HillClimbTrainer + makeTrainer()
│   ├── Training.h / .cpp   # TrainingSession: loop multi-mapa, curriculum, augmentation, val/test
│   ├── TrainingMath.h/.cpp # CVaR, rank-CVaR, z-score e agregações de fitness
│   ├── Game.h / .cpp       # reset/step (RL), tick batch, thread pool
│   ├── Renderer.h / .cpp   # ÚNICA camada SFML: pista, carros, HUD, gráfico fitness
│   ├── HumanController.h/.cpp  # Leitura de teclado → Action (depende de SFML)
│   └── main.cpp            # Parse de args, dispatch para todos os modos
├── tools/
│   ├── check_map_overlap.py  # Detecta auto-sobreposição do ribbon da pista; salva PNG opcional
│   └── watch_training.py     # Plota curvas de fitness/validação ao vivo durante o treino
└── tests/
    └── test_main.cpp       # Testes: Vec2, geometria, NN, determinismo, GA, Trainers, TrainingSession, reward
```
