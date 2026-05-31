# Arquitetura — Racing ML Sim

Este documento descreve em profundidade como o sistema funciona internamente: módulos, algoritmos, fluxo de dados, decisões de design e extensibilidade.

---

## Sumário

1. [Visão geral](#visão-geral)
2. [Princípio central: núcleo sem SFML](#princípio-central-núcleo-sem-sfml)
3. [Módulos](#módulos)
   - [core/ — tipos e matemática](#core--tipos-e-matemática)
   - [Track — pista e geometria](#track--pista-e-geometria)
   - [Sensor — raycasting](#sensor--raycasting)
   - [Car — física e estado](#car--física-e-estado)
   - [AIController — interface de agente](#aicontroller--interface-de-agente)
   - [NeuralNetwork — MLP e serialização](#neuralnetwork--mlp-e-serialização)
   - [GeneticAlgorithm — neuroevolução](#geneticalgorithm--neuroevolução)
   - [Game — loop de simulação](#game--loop-de-simulação)
   - [Renderer — camada SFML](#renderer--camada-sfml)
   - [HumanController — entrada de teclado](#humancontroller--entrada-de-teclado)
   - [main.cpp — entrypoint e dispatch](#maincpp--entrypoint-e-dispatch)
   - [tools/ — utilitários externos](#tools--utilitários-externos)
4. [Fluxo de dados por tick](#fluxo-de-dados-por-tick)
5. [Espaço de observação e ação](#espaço-de-observação-e-ação)
6. [Função de reward](#função-de-reward)
7. [Algoritmos geométricos](#algoritmos-geométricos)
   - [Geração de bordas](#geração-de-bordas)
   - [Raycast raio-segmento](#raycast-raio-segmento)
   - [Detecção de colisão](#detecção-de-colisão)
   - [Progresso monotônico](#progresso-monotônico)
8. [Timestep fixo e loop de render](#timestep-fixo-e-loop-de-render)
9. [Multithreading e determinismo](#multithreading-e-determinismo)
10. [Serialização binária da rede neural](#serialização-binária-da-rede-neural)
11. [Grafo de dependências entre módulos](#grafo-de-dependências-entre-módulos)
12. [Decisões de design e trade-offs](#decisões-de-design-e-trade-offs)
13. [Como estender o projeto](#como-estender-o-projeto)

---

## Visão geral

```
┌─────────────────────────────────────────────────────────────┐
│                         main.cpp                            │
│   Parse args → monta SimConfig → despacha modo              │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
          headless/benchmark          modo janela
               │                          │
               ▼                          ▼
         ┌──────────┐            ┌──────────────────┐
         │  Game    │◄───────────│   Renderer       │
         │  (loop)  │            │   (SFML, apenas) │
         └────┬─────┘            └──────────────────┘
              │
    ┌─────────┼──────────┐
    │         │          │
    ▼         ▼          ▼
  Track     Car[]    AIController[]
  (pista)  (física)  (agentes)
    │         │
    ▼         ▼
  Sensor   NeuralNetwork / HumanController / qualquer impl
```

O `Game` é o coração: guarda o `Track`, o vetor de `Car`s e o vetor de `AIController`s. A cada `tick()`, para cada carro ativo:

1. `Car::observe()` monta o vetor de observação (26 floats)
2. `AIController::decide()` retorna uma `Action` (2 floats)
3. `Car::applyAction()` integra a física com DT fixo
4. `Car::stepDone()` atualiza sensores, progresso, reward e condições de `done`

O `Renderer` lê o estado do `Game` passivamente — nunca escreve nele.

---

## Princípio central: núcleo sem SFML

A regra mais importante da arquitetura: **`#include <SFML/...>` só pode aparecer em `Renderer.cpp`, `Renderer.h`, `HumanController.cpp` e `HumanController.h`**.

Isso é verificável com:

```bash
grep -r "#include <SFML" src/ | grep -v "Renderer\|HumanController"
# Sem saída = correto
```

Consequências:
- A simulação compila e roda sem display (headless, testes, CI)
- `Track`, `Car`, `Sensor`, `Game`, `NeuralNetwork` usam só STL e `Vec2` próprio
- Trocar SFML por outra lib de render não exige tocar na lógica de simulação

---

## Módulos

### `core/` — tipos e matemática

**`TrackGen.h/.cpp`** — geração procedural de pistas e augmentation, sem SFML:
- `generateLoop(seed, params)` — loop fechado via soma de harmônicos de baixa frequência; validado e re-sorteado até `maxAttempts`; fallback elíptico determinístico
- `mirrorX(base)` — espelha em torno do eixo vertical do bounding-box
- `reverse(base)` — inverte a ordem dos waypoints
- `scaleWidth(base, factor)` — escala `track_width` pelo fator dado

**`Vec2.h`** — vetor 2D header-only, sem dependências:

```
Vec2{x, y}
  + - * /         → operações elemento a elemento
  dot(o)          → produto escalar
  length()        → magnitude
  normalized()    → vetor unitário (retorna {0,0} se length < 1e-9)
  rotated(rad)    → rotaciona no sentido horário (convenção y-down)
  perpendicular() → normal à esquerda em y-down: {1,0} → {0,-1} (para cima)
```

A convenção y-down (origem no canto superior esquerdo, y cresce para baixo) é a convenção de tela padrão. `perpendicular()` retorna `{y, -x}` para que "esquerda" signifique "para cima na tela" quando andando para a direita.

**`Constants.h`** — todas as constantes da simulação como `constexpr`. Mudar `NUM_RAYS` ou `NUM_LOOKAHEADS` aqui muda o tamanho do `Observation` em toda a cadeia automaticamente (via `OBS_SIZE = NUM_RAYS + 3 + 2 × NUM_LOOKAHEADS`).

**`Types.h`** — os tipos de fronteira do ambiente de RL:

```cpp
using Observation = std::array<float, OBS_SIZE>; // 26 floats

struct Action      { float throttle, steering; }; // ambos ∈ [-1, 1]
struct StepResult  { Observation next; float reward; bool done; };
struct RewardConfig{ float w_progress, w_speed, w_checkpoint, w_finish, w_time,
                           w_crash, w_reverse, w_regress, w_curve; };
struct SimConfig   { int population; unsigned seed; bool headless;
                     std::string map; RewardConfig reward; int threads; };
```

---

### `Track` — pista e geometria

**Responsabilidade:** carregar o circuito de um JSON, gerar as bordas e responder a consultas geométricas (raycast, colisão, progresso).

**Carregamento JSON** (`nlohmann/json` via FetchContent):
- Valida campos obrigatórios: `waypoints`, `track_width`, `closed`
- Lança `std::runtime_error` com mensagem descritiva em qualquer divergência de schema
- Suporta obstáculos opcionais: `circle` (pos + radius) e `rect` (pos + size)

**Geração de bordas** (`buildBorders()`):

Para cada waypoint `i`, calcula a direção local como:
```
dir = normalize(waypoints[i+1] - waypoints[i-1])   // pista fechada
dir = normalize(waypoints[1] - waypoints[0])         // endpoint inicial aberto
```

A borda esquerda e direita são os offsets perpendiculares:
```
leftBorder[i]  = waypoints[i] + dir.perpendicular() * (track_width / 2)
rightBorder[i] = waypoints[i] - dir.perpendicular() * (track_width / 2)
```

Para pistas fechadas, o último ponto é uma cópia do primeiro, fechando o polígono.

**Estrutura de dados:**
```cpp
std::vector<Vec2> waypoints_;    // linha central (N pontos)
std::vector<Vec2> leftBorder_;   // borda esquerda (N+1 para fechadas)
std::vector<Vec2> rightBorder_;  // borda direita  (N+1 para fechadas)
std::vector<Obstacle> obstacles_;
```

---

### `Sensor` — raycasting

**Responsabilidade:** dado a posição e ângulo do carro, lançar `NUM_RAYS` raios e retornar distâncias normalizadas.

Os raios são distribuídos uniformemente em um leque de **180°** centrado na direção do carro:

```
ângulo_inicial = angle - π/2       // extremo esquerdo
passo          = π / (NUM_RAYS-1)  // distribui igualmente
ray_i_angle    = ângulo_inicial + passo * i
```

Cada raio chama `Track::raycast()` e normaliza: `reading[i] = dist / RAY_MAX_LEN ∈ [0, 1]`.

O Sensor também salva os **pontos de impacto** (`hitPoints_`) para o `Renderer` desenhar as linhas de debug.

---

### `Car` — física e estado

**Responsabilidade:** manter o estado físico de um agente e avançar um passo de simulação.

**Estado público:**
```cpp
Vec2  pos;           // posição no mundo (pixels)
float angle;         // orientação em radianos
float speed;         // velocidade escalar (px/s), negativa = ré
float fitness;       // recompensa acumulada no episódio
float idleTime;      // segundos consecutivos sem movimento
float episodeTime;   // segundos desde o início do episódio
bool  done;
DoneReason doneReason; // None | Collision | Timeout | Stall | Completed
ProgressState progState;
Sensor sensor;
```

**Física (`applyAction`):**

```
accelForce = throttle > 0 ? throttle * ACCEL * DT
                           : |throttle| * BRAKE * DT
speed += accelForce
speed *= DRAG              // atrito por tick (0.98^60 ≈ 0.30 em 1s)
speed = clamp(speed, -MAX_SPEED, MAX_SPEED)

steerRate = steering * MAX_STEER * (|speed| / MAX_SPEED)
angle += steerRate * DT    // steering escalado pela velocidade atual

pos.x += cos(angle) * speed * DT
pos.y += sin(angle) * speed * DT
```

O steering escalado pela velocidade (car model simplificado) impede que o carro gire no lugar a zero de velocidade.

**Condições de `done` (verificadas em `stepDone`, nessa ordem):**

1. `progState.nextWp > totalWps` → **Completed** + bônus `w_finish`
2. `episodeTime >= EPISODE_TIMEOUT` → **Timeout**
3. `idleTime >= STALL_TIMEOUT` → **Stall**
4. `!track.isInsideTrack(pos)` → **Collision** - penalidade `w_crash`

---

### `AIController` — interface de agente

```cpp
class AIController {
public:
    virtual ~AIController() = default;
    virtual Action decide(const Observation&) = 0;
    virtual void reset() {}  // chamado no início de cada episódio
};
```

Interface mínima e intencional. Qualquer algoritmo de ML ou controle manual implementa apenas `decide()`. O `Game` não sabe o que está dentro dos controladores — é um polimorfismo puro.

Implementações existentes:
- `NeuralNetworkController` — wraps uma `NeuralNetwork`
- `HumanController` — lê `sf::Keyboard` (só disponível fora de `HEADLESS_ONLY`)

---

### `NeuralNetwork` — MLP e serialização

**Rede feedforward** com topologia parametrizável, ativação `tanh` em todas as camadas (saída já em (-1, 1), compatível com o espaço de ação).

**Topologia padrão:** `{26, 32, 2}` — 26 entradas (OBS_SIZE), 32 ocultos (configurável via `--hidden`), 2 saídas (throttle + steering).

**Armazenamento de pesos:**
```
weights_[l]  → matriz row-major [out][in] da camada l
biases_[l]   → vetor de bias da camada l+1
```

**Forward pass:**
```
cur = input
for cada camada l:
    for cada neurônio o:
        sum = biases_[l][o] + Σ(weights_[l][o*in + i] * cur[i])
        next[o] = tanh(sum)
    cur = next
return cur
```

**Inicialização:** Xavier-uniform: pesos ∈ `[-sqrt(6/(in+out)), +sqrt(6/(in+out))]`, biases = 0.

**Interface para o GA:**
```cpp
std::vector<float> getWeights();       // vetor flat: pesos e biases por camada
void setWeights(const std::vector<float>&);
```

**Serialização binária versionada:**

```
offset  tamanho   conteúdo
0       4 bytes   magic "RNNW"
4       4 bytes   version (uint32, atualmente 1)
8       4 bytes   nlayers (uint32)
12      4*nlayers tamanhos de cada camada
...     4 bytes   nweights (uint32)
...     4*nweights pesos float32
```

`load()` e `loadFromBuffer()` validam magic, version e topologia antes de ler os pesos; lançam `std::runtime_error` em qualquer divergência.

---

### `GeneticAlgorithm` — neuroevolução

Interface completa com implementação funcional mínima. Projetado para ser substituído por NEAT, CMA-ES ou outro algoritmo sem mudar a interface.

**Ciclo de uma geração:**

```
initPopulation(n, weightCount, seed)
│
├── cria n Genome{weights aleatórios, fitness=0}
│
loop de gerações:
│
├── simulação roda → setFitness(i, fitness_i) para cada carro
│
└── evolve():
    ├── elitismo: top 10% sobrevive sem alteração
    ├── para os demais:
    │   ├── tournamentSelect() × 2 → parents A e B
    │   ├── crossover de ponto único → filho
    │   └── mutação gaussiana (rate=10%, sigma=0.3)
    └── geração++
```

**Seleção por torneio (`k=3`):** sorteia 3 índices aleatórios da população, retorna o de maior fitness. Equilibra pressão seletiva e diversidade.

**RNG:** usa LCG (`lcg()`) próprio sem deps externas, semeado com `seed XOR generation`. Determinístico para qualquer configuração.

---

### `Game` — loop de simulação

O `Game` é o **único ponto de contato** entre a simulação e o código externo. Expõe duas interfaces:

**Interface RL (single-agent):**
```cpp
Observation reset()              // reposiciona todos os carros, retorna obs[0]
StepResult  step(const Action&)  // 1 tick do carro 0, retorna {obs, reward, done}
```

**Interface batch (neuroevolução):**
```cpp
void tick()                         // avança TODOS os carros 1 tick (multithread)
bool episodeDone()                  // true quando todos os carros estão done
std::vector<float> fitnesses()      // fitness por carro no fim do episódio
double runHeadlessEpisode()         // loop completo, retorna segundos de wall-clock
```

**Multithreading em `tick()`:**

```
se numThreads <= 1 ou population < 64:
    updateRange(0, N)    // single thread
senão:
    chunk = ceil(N / numThreads)
    para cada thread t:
        lança thread → updateRange(t*chunk, min((t+1)*chunk, N))
    join de todas as threads
```

Cada thread opera em uma fatia disjunta do vetor de carros. A `Track` é read-only durante o episódio. Carros são independentes entre si (sem colisão carro-a-carro). Portanto, **zero locks no caminho quente**.

---

### `Renderer` — camada SFML

**Responsabilidade exclusiva:** ler o estado do `Game` e desenhar na janela. Nunca modifica o estado da simulação.

**O que renderiza:**
- Bordas esquerda e direita da pista (linhas brancas)
- Waypoints (círculos azuis semitransparentes)
- Obstáculos (círculos/retângulos vermelhos)
- Carros (retângulos 20×10px, amarelo para o carro 0, verde para os demais)
- Raios de sensor do carro 0 (linha colorida: verde = longe, vermelho = perto)
- HUD com velocidade, tempo de episódio, progresso, número de carros

**Performance:** em populações grandes (>200), renderiza no máximo 200 carros para manter o framerate.

**API SFML 3 usada:**
```cpp
sf::VideoMode({900u, 700u})               // tamanho da janela
window.pollEvent()                         // retorna std::optional<sf::Event>
e->is<sf::Event::Closed>()                // verificação de evento tipado
sf::Keyboard::isKeyPressed(Key::Up)       // polling de teclado
sf::Font::openFromFile(path)              // carrega fonte
sf::Text(font, string, size)              // texto com fonte obrigatória no ctor
sf::radians(angle)                        // ângulo para rotação de shape
```

---

### `HumanController` — entrada de teclado

Polling direto de `sf::Keyboard` a cada tick. Retorna a `Action` correspondente às teclas pressionadas.

Compilado condicionalmente: em builds com `-DHEADLESS_ONLY` (usado em `racing_tests`), um stub é compilado no lugar, eliminando a dependência de SFML.

```cpp
#ifndef HEADLESS_ONLY
// implementação real com sf::Keyboard
#else
Action HumanController::decide(const Observation&) { return {}; }
#endif
```

---

### `tools/` — utilitários externos

**`check_map_overlap.py`** — detecta auto-sobreposição do ribbon da pista (bordas que se cruzam). Replica a geometria C++ (Catmull-Rom + miter offset) em Python. Uso:
```bash
python3 tools/check_map_overlap.py maps/meu_mapa.json [saida.png]
```

**`watch_training.py`** — plota ao vivo as curvas de fitness de treino, validação e teste a partir do CSV gerado com `--log-csv`. Relê o arquivo a cada segundo.

---

### `main.cpp` — entrypoint e dispatch

Parse linear de argumentos → monta `SimConfig` → despacha para um de três caminhos:

```
--benchmark  →  Game::runBenchmark(cfg)      // estático, imprime tempo e sai
--headless   →  Game g(cfg); g.runHeadlessEpisode()
(default)    →  Game + Renderer + loop de acumulador
```

No modo janela, substitui os controladores: carro 0 → `HumanController`, demais → `NeuralNetworkController` com seeds derivadas de `cfg.seed + 1`.

---

## Fluxo de dados por tick

```
┌─────────────────────────────────────────────────────────────┐
│  Game::tick()  [eventualmente paralelo por chunk de carros]  │
│                                                              │
│  Para cada carro i não-done:                                 │
│                                                              │
│  1. obs = Car::observe(track)                                │
│     ├── lê sensor.readings() [13 floats]                     │
│     ├── calcula speed_norm [1 float]                         │
│     ├── calcula lateral_offset [1 float]                     │
│     ├── calcula heading_error [1 float]                      │
│     └── 5 × (curvatura, speed_excess) [10 floats]           │
│         → Observation[26]                                    │
│                                                              │
│  2. act = controllers[i]->decide(obs)                        │
│     └── NeuralNetwork::forward(obs) → {throttle, steering}  │
│                                                              │
│  3. Car::applyAction(act)                                    │
│     ├── atualiza speed (accel/brake + drag)                  │
│     ├── atualiza angle (steering escalado por speed)         │
│     └── integra pos (Euler explícito)                        │
│                                                              │
│  4. Car::stepDone(track, reward_cfg)                         │
│     ├── atualiza episodeTime                                 │
│     ├── Sensor::update(pos, angle, track)  [13 raycasts]     │
│     ├── Track::progressAt(pos, progState)  [delta progresso] │
│     ├── calcula reward do tick                               │
│     ├── verifica done: Completed / Timeout / Stall / Crash   │
│     └── acumula fitness                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Espaço de observação e ação

### Observação — 26 floats

| Índice | Campo | Intervalo | Descrição |
|---|---|---|---|
| 0..12 | `ray_0` … `ray_12` | [0, 1] | Distância normalizada de cada raio ao obstáculo/borda mais próximo. 0 = borda imediata, 1 = nada até `RAY_MAX_LEN` |
| 13 | `speed_norm` | [0, 1] | `|speed| / MAX_SPEED` |
| 14 | `lateral_offset` | [-1, 1] | Offset lateral em relação à linha central (positivo = à direita da pista) |
| 15 | `heading_error` | [-1, 1] | Ângulo do carro vs tangente da pista, normalizado por π |
| 16,17 | lookahead 1 | [-1,1] × [-1,1] | (curvatura assinada, speed_excess) a 30px de arco à frente |
| 18,19 | lookahead 2 | — | 60px |
| 20,21 | lookahead 3 | — | 100px |
| 22,23 | lookahead 4 | — | 160px |
| 24,25 | lookahead 5 | — | 240px |

`speed_excess` = (velocidade atual − velocidade segura para o raio de curvatura) / MAX_SPEED. Positivo significa que o carro está rápido demais para a curva à frente.

Os raios são distribuídos em leque de 180° à frente:
```
ray_0  → aponta 90° à esquerda
ray_6  → aponta direto à frente
ray_12 → aponta 90° à direita
```

### Ação — 2 floats

| Campo | Intervalo | Significado |
|---|---|---|
| `throttle` | [-1, 1] | > 0: acelera, < 0: freia/ré |
| `steering` | [-1, 1] | < 0: esquerda, > 0: direita |

A saída `tanh` da rede neural cai naturalmente em (-1, 1), casa direto com o espaço de ação.

---

## Função de reward

O fitness final de um episódio é:

```
fitness = w_progress  × maxProgress                    (fração do lap em arco, ∈ [0,1])
        + w_speed     × ∫ (|v| / MAX_SPEED) dt         (acumulado enquanto avança)
        + w_checkpoint× Σ bônus por waypoint cruzado    (escalado pela curvatura local)
        − w_reverse   × ∫ (−v) dt                      (acumulado quando speed < 0)
        − w_regress   × ∫ (maxProgress − progress) dt  (acumulado ao regredir)
        − w_curve     × ∫ |curvature| × |v| dt          (desativado por padrão)
        + (se Completed) w_finish + w_time × (timeout − episodeTime)
        − (se Collision) w_crash
```

**maxProgress** é a maior fração de arco da centerline que o carro alcançou no episódio — não decresce se o carro recuar. É a métrica primária de generalização reportada no terminal e na validação.

Defaults: `w_progress=200, w_speed=0.3, w_checkpoint=5, w_finish=100, w_time=2, w_crash=15, w_reverse=0.5, w_regress=2, w_curve=0`.

---

## Algoritmos geométricos

### Geração de bordas

Para cada waypoint `i` numa pista fechada:
```
prev = waypoints[(i - 1 + N) % N]
next = waypoints[(i + 1) % N]
dir  = normalize(next - prev)           // direção suavizada no waypoint
perp = dir.perpendicular()              // {dir.y, -dir.x} (left normal, y-down)
leftBorder[i]  = waypoints[i] + perp * (track_width / 2)
rightBorder[i] = waypoints[i] - perp * (track_width / 2)
```

Usar `next - prev` (ao invés de apenas `next - curr`) suaviza a direção nos cantos, evitando bordas com "quebras" abruptas.

### Raycast raio-segmento

Dado raio `origin + t * dir` e segmento `a + u * (b-a)`:

```
ab  = b - a
det = dir.x * ab.y - dir.y * ab.x      // det = dir × ab (cross 2D)
ao  = origin - a

t = -(ao.x * ab.y - ao.y * ab.x) / det // parâmetro ao longo do raio
u = -(ao.x * dir.y - ao.y * dir.x) / det // parâmetro ao longo do segmento

válido se: t >= 0 && u ∈ [0, 1]
```

Retorna `t` (distância) se válido, `+∞` caso contrário.

Para **obstáculos circulares**, usa a fórmula da interseção raio-círculo:
```
oc = origin - center
disc = (2 * oc·dir)² - 4 * (oc·oc - r²)
t = (-2*oc·dir - √disc) / 2            // menor t positivo
```

Para **obstáculos retangulares (AABB)**, testa os 4 segmentos das bordas do retângulo.

### Detecção de colisão

`isInsideTrack(pos)` percorre todos os segmentos do centro da pista (média entre bordas esquerda e direita) e verifica se `pos` está a menos de `track_width/2` de algum segmento via projeção:

```
ab = b - a
t  = clamp((p - a)·ab / |ab|², 0, 1)  // projeção do ponto no segmento
closest = a + ab * t
inside = |p - closest| < track_width / 2
```

### Progresso por arco

`ProjectionState` guarda `segIdx` (segmento do centerline denso mais próximo), `t` (interpolação dentro do segmento) e `arcLen` (comprimento de arco acumulado dentro do lap atual).

A cada tick, `Track::updateProjection` faz busca local no centerline Catmull-Rom densificado (`CENTERLINE_SUBSEGMENTS=10` sub-segmentos por waypoint de design):

```
arcLen = Σ distâncias ao longo dos segmentos até a projeção do ponto atual
maxProgress = max(maxProgress, arcLen / totalArcLength)  // monotônico
```

`maxProgress ∈ [0, 1]` é a fração do lap coberta — nunca decresce, mesmo se o carro recuar. Com `--random-spawn`, o spawn é posicionado em um ponto aleatório do arco e o progresso é medido a partir dele.

---

## Timestep fixo e loop de render

O modo janela usa um **acumulador de tempo** para desacoplar a frequência de render (limitada pelo framerate do monitor) do timestep fixo da simulação (60 Hz):

```cpp
auto prev = Clock::now();
float accumulator = 0.f;

while (window.isOpen()) {
    float frame_dt = (Clock::now() - prev).count();
    prev = Clock::now();
    accumulator += frame_dt;

    while (accumulator >= DT) {     // DT = 1/60 s
        game.tick();
        accumulator -= DT;
        if (game.episodeDone()) { game.reset(); accumulator = 0; break; }
    }

    renderer.render(game);          // 1 render por frame, independente de quantos ticks
}
```

Isso garante que a **física é determinística** independente do framerate. Se o monitor roda a 120Hz, a simulação ainda avança a 60 ticks/segundo. Se o sistema estiver lento, acumula e processa múltiplos ticks por frame.

---

## Multithreading e determinismo

**Por que é paralelo:** carros são independentes — cada um só lê seu próprio estado + o `Track` (imutável durante o episódio). Não há redução de floats entre carros.

**Por que é determinístico:** cada carro tem seu próprio RNG semeado durante a construção do `Game`. A ordem de execução dos carros dentro de um tick não importa (não há dependência entre eles), portanto o resultado por carro é idêntico com qualquer número de threads.

**Implementação:** threads são criadas e destruídas a cada `tick()`. Não é um thread pool com workers persistentes — a overhead de criação/join é visível em populações pequenas, mas insignificante para N ≥ 64 (limiar em que o multithreading é ativado).

**Garantia testada:** o teste `test_car_determinism_multithread` roda 100 carros por 300 ticks com single-thread e com `hardware_concurrency` threads e verifica que o fitness de cada carro é idêntico.

---

## Serialização binária da rede neural

Formato `RNNW v1`:

```
[0..3]   "RNNW"           magic (4 bytes ASCII)
[4..7]   1                version (uint32 little-endian)
[8..11]  nlayers          número de camadas na topologia (uint32)
[12..]   sizes[]          tamanho de cada camada (nlayers × uint32)
[...]    nweights         total de floats (uint32)
[...]    weights[]        pesos e biases em ordem: camada 0 pesos, camada 0 biases, camada 1 pesos, ...
```

`load()` valida em sequência:
1. magic == "RNNW" (senão: "invalid magic")
2. version == 1 (senão: "unsupported version N")
3. topologia do arquivo == topologia da rede (senão: "topology mismatch")

Permite detectar arquivos corrompidos, de versões antigas ou de redes com arquitetura diferente antes de qualquer leitura de pesos.

---

## Grafo de dependências entre módulos

```
                    ┌─────────────┐
                    │  main.cpp   │
                    └──────┬──────┘
                           │ usa
              ┌────────────┼──────────────────┐
              ▼            ▼                  ▼
         ┌────────┐  ┌──────────┐     ┌──────────────┐
         │  Game  │  │ Renderer │     │ HumanCtrl    │
         └───┬────┘  └────┬─────┘     └──────────────┘
             │             │ lê (read-only)  │ SFML
    ┌────────┼────────┐    │                 │
    ▼        ▼        ▼    │                 │
  Track    Car[]  AICtrl[] │                 │
    │        │        │    │                 │
    │        ▼        ▼    │                 │
    │     Sensor   NeuralNetwork             │
    │        │     GeneticAlgorithm          │
    │        │                               │
    ▼        ▼                               │
  core/Vec2, core/Constants, core/Types ◄───┘
  (sem SFML, sem deps externas)
```

Regras:
- `core/` não depende de ninguém
- `Track`, `Car`, `Sensor`, `NeuralNetwork`, `GeneticAlgorithm` só dependem de `core/`
- `Game` depende de tudo exceto `Renderer` e `HumanController`
- `Renderer` e `HumanController` são as únicas folhas que incluem SFML
- `main.cpp` é o único ponto que conecta tudo

---

## Decisões de design e trade-offs

| Decisão | Alternativa | Motivo da escolha |
|---|---|---|
| SFML 3 (não 2) | SFML 2 | API mais moderna, suporte ativo, eventos type-safe com `is<>()` |
| `nlohmann/json` via FetchContent | Arquivo local, vcpkg | Zero configuração manual; json é editável à mão |
| Vec2 próprio | `sf::Vector2f` em todo lugar | Desacopla lógica de SFML; testes headless |
| Serialização binária da NN | JSON/texto | Compacto e rápido para salvar/carregar populações de milhares de redes |
| Threads criadas por tick | Thread pool persistente | Simplicidade; threshold de N≥64 limita overhead para populações pequenas |
| Carros sem colisão mútua | Colisão carro-a-carro | Mantém carros independentes → paralelismo sem locks; revisável no futuro |
| Progresso monotônico por waypoint | Distância percorrida | Robusto para pistas com curvas; não decai se o carro recua |
| GA como stub funcional | GA completo (NEAT) | Escopo controlado; interface pronta para implementação futura |
| Compilação condicional `HEADLESS_ONLY` | Link opcional | Permite compilar os testes sem SFML, sem mudança de código |

---

## Generalização: diversidade, anti-memorização e seleção por held-out

Para evitar que a neuroevolução **decore** os mapas de treino, há quatro frentes opcionais
(todas desligadas por padrão — o caminho determinístico dos testes é preservado bit-a-bit).

**`core/TrackGen` (sem SFML)** — opera sobre `TrackData` (descrição de pista desacoplada de
arquivo) que o `Track` constrói em memória pela mesma pipeline da carga JSON (`Track::loadData`
+ `Track(const TrackData&)`):
- **Augmentation** (`mirrorX`, `reverse`, `scaleWidth`): variantes estáticas de cada mapa base,
  anexadas ao conjunto de treino via `--augment`. Jitter de largura é variante pré-construída
  (não rebuild por episódio), mantendo o caminho quente barato.
- **Geração procedural** (`generateLoop`, semeada): raio como soma de harmônicos de baixa
  frequência → loop *star-convex* (sem auto-interseção) e suave. Cada candidato é validado
  (arco > 0, sem auto-overlap do ribbon, footprint do carro dentro ao longo de toda a pista,
  e spawn dentro) com re-sorteio até `maxAttempts`; fallback elíptico determinístico.

**Randomização por episódio** (`Game::EpisodeConfig`, default reproduz o episódio legado):
- **Spawn aleatório** ao longo do centerline. O progresso passa a ser medido **relativo ao
  spawn** (`Car::spawnArcFrac`): com `spawnArcFrac == 0` (padrão) é idêntico ao atual; com spawn
  no meio da pista o carro não ganha progresso grátis no tick 0 e precisa completar uma volta a
  partir de onde nasceu.
- **Ruído de sensor** aplicado às leituras de raio no loop de `simulateEpisode` (não no `Car`).
- **N episódios por avaliação** (`TrainingSession::evalCar`), agregados por média/mínimo, com
  seed por episódio derivada de `(seed, geração, mapa, carro, episódio)`.

**Seleção por held-out** (3 conjuntos: treino / validação / teste): com `--select-by-val`,
`best.rnnw` é o genoma com **melhor progresso de validação** entre os top-K do treino
(`selectChampionByVal`), não o melhor fitness de treino. O conjunto de **teste** (`--test-maps`)
é só relatório (`test_log.csv`), nunca usado na seleção — preserva uma métrica não-contaminada.
Validação/teste rodam com episódio **determinístico** (spawn fixo, sem ruído) para métricas
comparáveis.

---

## Como estender o projeto

### Adicionar um novo algoritmo de ML

```cpp
// src/MinhaIA.h
#include "AIController.h"

class MinhaIA : public AIController {
public:
    Action decide(const Observation& obs) override { /* ... */ }
    void   reset() override { /* ... */ }
};
```

Nenhum outro arquivo precisa ser alterado. Injete via `game.setControllers(...)`.

### Adicionar um novo tipo de obstáculo

1. Adicione um valor ao `enum class Obstacle::Type` em `Track.h`
2. Carregue o JSON em `Track::Track()` em `Track.cpp`
3. Adicione o raycast em `Track::raycast()`
4. Adicione o desenho em `Renderer::drawTrack()`

### Aumentar o número de raios ou mudar o OBS_SIZE

Mude `NUM_RAYS` em `Constants.h`. `OBS_SIZE = NUM_RAYS + 3` atualiza automaticamente o tipo `Observation`. A topologia da NN padrão (`{OBS_SIZE, 8, 2}`) também pega o novo valor. Recompile.

### Adicionar um novo modo de execução

Em `main.cpp`, adicione um novo argumento (`--evolve`, por exemplo) e a lógica de dispatch correspondente. O `Game` já expõe tudo que é necessário.

### Trocar a renderização (ex.: SDL, raylib)

Reescreva apenas `Renderer.cpp/h` e `HumanController.cpp/h`. O restante do projeto não muda.
