# Bem-vindo ao
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \   __\   __\  | _/ __ \ /  ___//    \__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\_____>
#
# COBRA PARA DUELOS 1v1 — versão 8: espaço temporal, cauda dinâmica e cerco.
#
# Estratégia: a cada turno a cobra SIMULA o jogo à frente (minimax com poda
# alpha-beta e aprofundamento iterativo): ela escolhe a jogada cujo PIOR cenário
# (a melhor resposta do rival) seja o melhor. Os estados são avaliados por
# território (Voronoi), tamanho, comida, fome, espaço livre e cerco.
# A busca respeita um orçamento de tempo e nunca estoura o limite do jogo.
# A v8 adiciona ocupação temporal: casas ocupadas hoje podem se tornar
# transitáveis nos próximos turnos conforme os corpos avançam.
#
# Organização deste arquivo (cada seção era um módulo):
#   1. Configuração (pesos, tempo, profundidade)      6. Território (Voronoi)
#   2. Tipos internos (Snake, Board, State)           7. Avaliação do estado
#   3. Regras do tabuleiro (simulação de um turno)    8. Busca (minimax / alpha-beta)
#   4. Segurança (filtros de morte certa)             9. Estratégia (choose_move)
#   5. Espaço (flood fill) e caminhos (BFS / A*)     10. Funções chamadas pelo jogo
# Documentação: https://docs.battlesnake.com

from __future__ import annotations

import heapq
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
# O runtime Python da Lambda deixa o logger raiz em WARNING: sem esta linha
# as jogadas nao aparecem no CloudWatch.
logger.setLevel(logging.INFO)


# =============================================================================
# 1. Configuração (pesos, tempo, profundidade)
# =============================================================================

# ----------------------------- Tempo e busca ---------------------------------
TIME_BUDGET_FRACTION = 0.42   # fração do timeout do jogo usada na busca (resto: rede + Lambda)
MIN_BUDGET_S = 0.05          # nunca busca por menos que isso
MAX_DEPTH = 12               # profundidade máxima (em turnos completos); o tempo costuma parar antes
BRANCHED_ENEMIES = 2         # só os N rivais mais próximos ramificam; os demais seguem jogada padrão

# ----------------------------- Regras ----------------------------------------
DEFAULT_HAZARD_DAMAGE = 14   # dano extra por turno em zona de hazard (royale)

# ----------------------------- Notas terminais -------------------------------
WIN_SCORE = 1_000_000.0      # vitória (somada à profundidade restante: vitória rápida vale mais)
DRAW_SCORE = -100_000.0      # todos morrem juntos: melhor que perder, pior que viver

# ----------------------------- Pesos da avaliação ----------------------------
# Em 1v1, ganhar espaço e reduzir as saídas do rival importa mais do que
# simplesmente buscar comida ou ficar perto do centro.
W_TERRITORY = 2.4
W_LENGTH = 4.0
LENGTH_CAP = 6
W_MOBILITY = 9.0
W_ENEMY_MOBILITY = 11.0
W_ROBUST_EXITS = 18.0
W_ENEMY_ROBUST_EXITS = 20.0
W_TAIL = 15.0
W_ENEMY_TRAPPED = 55.0
W_ATTACK_PRESSURE = 12.0
W_FOOD_TERRITORY = 2.5
W_HUNGER = 10.0
HUNGER_START = 60
STARVE_PENALTY = 550.0
TRAPPED_MARGIN = 7
W_CENTER = 0.15
W_OPPONENT_CUT = 24.0
W_WINNING_TENSION = 8.0
W_HEALTH = 0.9
W_TAIL_DISTANCE = 10.0
W_FOOD_RACE = 2.4
W_FORCING = 75.0
W_ENDGAME_CUT = 12.0

# ------------------------- Espaço temporal (v8) -----------------------------
W_TEMPORAL_SPACE = 8.0
W_TEMPORAL_SPACE_DIFF = 7.0
W_TEMPORAL_TAIL = 18.0
W_TEMPORAL_ESCAPE = 10.0
W_CORRIDOR_OPENING = 7.0


# Checagem tática fora do minimax: detecta mate em 1 ou mate forçado em 2.
# Como o duelo 1v1 possui somente um adversário, o custo é aceitável em 11x11.
FORCED_WIN_CHECK = True
FORCED_WIN_BUDGET_FRACTION = 0.22  # máximo do orçamento para a checagem tática


# ----------------------------- Orientação da estratégia ----------------------
LATENCY_ALERT = 0.8
LATENCY_REDUCTION = 0.6
ROOT_ASPIRATION = 30.0


# =============================================================================
# 2. Tipos internos
# =============================================================================

Point = tuple[int, int]   # (x, y); "up" aumenta y, "right" aumenta x


@dataclass(slots=True)
class Snake:
    id: str
    body: tuple[Point, ...]   # body[0] é a cabeça; tupla = imutável e compartilhável
    health: int

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass(slots=True)
class Board:
    width: int
    height: int
    food: frozenset = frozenset()
    hazards: frozenset = frozenset()
    hazard_damage: int = DEFAULT_HAZARD_DAMAGE


@dataclass(slots=True)
class State:
    board: Board
    snakes: list            # só cobras VIVAS (cobra eliminada sai da lista)
    me_id: str
    turn: int = 0
    _blocked: set | None = field(default=None, repr=False, compare=False)  # cache
    _free_at: dict | None = field(default=None, repr=False, compare=False)  # v8: liberação temporal

    def me(self) -> Snake | None:
        for s in self.snakes:
            if s.id == self.me_id:
                return s
        return None

    def enemies(self) -> list:
        return [s for s in self.snakes if s.id != self.me_id]


def _hazard_damage(api_state) -> int:
    """Lê hazardDamagePerTurn das regras (ruleset pode ser dict ou objeto)."""
    try:
        ruleset = api_state.game.ruleset
        settings = ruleset.get("settings") if isinstance(ruleset, dict) else getattr(ruleset, "settings", None)
        if isinstance(settings, dict):
            value = settings.get("hazardDamagePerTurn")
        else:
            value = getattr(settings, "hazardDamagePerTurn", None)
        if isinstance(value, int):
            return value
    except Exception:
        pass
    return DEFAULT_HAZARD_DAMAGE


def from_api(api_state) -> State:
    """Converte o State do template (Pydantic) para o State interno."""
    b = api_state.board
    snakes = [
        Snake(s.id, tuple((p.x, p.y) for p in s.body), s.health)
        for s in b.snakes
    ]
    me = api_state.you
    if not any(s.id == me.id for s in snakes):
        snakes.append(Snake(me.id, tuple((p.x, p.y) for p in me.body), me.health))
    board = Board(
        width=b.width,
        height=b.height,
        food=frozenset((f.x, f.y) for f in b.food),
        hazards=frozenset((h.x, h.y) for h in (getattr(b, "hazards", None) or [])),
        hazard_damage=_hazard_damage(api_state),
    )
    return State(board, snakes, me.id, getattr(api_state, "turn", 0))


# =============================================================================
# 3. Regras do tabuleiro
# =============================================================================

DIRECTIONS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
MOVES = tuple(DIRECTIONS)


@lru_cache(maxsize=8)
def neighbor_table(width: int, height: int) -> dict:
    """Tabela casa -> vizinhas dentro do tabuleiro (calculada uma vez por tamanho)."""
    table = {}
    for x in range(width):
        for y in range(height):
            cand = ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y))
            table[(x, y)] = tuple(p for p in cand if 0 <= p[0] < width and 0 <= p[1] < height)
    return table


def in_bounds(board: Board, p: Point) -> bool:
    return 0 <= p[0] < board.width and 0 <= p[1] < board.height


def step(p: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return (p[0] + dx, p[1] + dy)


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def just_ate(snake: Snake) -> bool:
    """Cobra que acabou de comer tem os dois últimos segmentos empilhados."""
    return len(snake.body) >= 2 and snake.body[-1] == snake.body[-2]


def blocked_cells(state: State) -> set:
    """Casas ocupadas por corpos. A cauda conta como livre (ela sai do lugar no
    próximo turno), exceto se a cobra acabou de comer.

    O resultado é guardado em cache no estado: NÃO modifique o conjunto devolvido
    (use `blocked | {x}` para criar uma cópia).
    """
    cached = state._blocked
    if cached is not None:
        return cached
    cells: set = set()
    for s in state.snakes:
        body = s.body
        if len(body) >= 2 and body[-1] != body[-2]:
            cells.update(body[:-1])
        else:
            cells.update(body)
    state._blocked = cells
    return cells


def free_at_cells(state: State) -> dict:
    """Mapa casa -> primeiro turno em que ela deixa de estar ocupada.

    A contagem é relativa ao estado atual: a cauda normal libera no turno 1,
    o segmento anterior à cauda no turno 2 e assim por diante. Se houve comida
    e a cauda está duplicada, a casa só fica livre no turno 2.
    """
    cached = state._free_at
    if cached is not None:
        return cached
    free_at: dict = {}
    for s in state.snakes:
        length = len(s.body)
        for i, seg in enumerate(s.body):
            release = length - i
            if release > free_at.get(seg, 0):
                free_at[seg] = release
    state._free_at = free_at
    return free_at


def temporal_bfs(state: State, start: Point, start_turn: int = 0,
                 reserve_start_until: int | None = None,
                 limit: int | None = None) -> dict[Point, int]:
    """BFS que permite atravessar casas quando elas ficam livres.

    A casa inicial é sempre permitida. Quando `reserve_start_until` é usado,
    ela é marcada como ocupada para impedir que a busca temporal atravesse
    novamente pela própria cabeça cedo demais.
    """
    table = neighbor_table(state.board.width, state.board.height)
    free_at = free_at_cells(state)
    blocked_until = dict(free_at)
    if reserve_start_until is not None:
        blocked_until[start] = max(blocked_until.get(start, 0), reserve_start_until)

    arrival: dict[Point, int] = {start: start_turn}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        t = arrival[cur] + 1
        for nb in table[cur]:
            if nb in arrival:
                continue
            if blocked_until.get(nb, 0) > t:
                continue
            arrival[nb] = t
            if limit is not None and len(arrival) >= limit:
                return arrival
            queue.append(nb)
    return arrival


def temporal_space_after_move(state: State, snake: Snake, move: str,
                              limit: int | None = None) -> tuple[int, dict[Point, int]]:
    """Espaço alcançável após um movimento, considerando liberação temporal."""
    target = step(snake.head, move)
    if not in_bounds(state.board, target):
        return 0, {}
    blocked = blocked_cells(state)
    if target in blocked:
        return 0, {}

    # A nova cabeça continua ocupada enquanto percorremos o corpo.
    reserve_until = snake.length + (1 if target in state.board.food else 0)
    arrival = temporal_bfs(state, target, 1, reserve_until, limit)
    return len(arrival), arrival


def temporal_tail_distance(state: State, snake: Snake) -> int | None:
    """Distância até a cauda usando as casas que se liberam ao longo do tempo."""
    if snake.length < 2 or just_ate(snake):
        return None
    arrival = temporal_bfs(
        state, snake.head, 0, snake.length,
        state.board.width * state.board.height,
    )
    return arrival.get(snake.body[-1])


def _keep_direction(s: Snake) -> str:
    if len(s.body) >= 2:
        dx = s.body[0][0] - s.body[1][0]
        dy = s.body[0][1] - s.body[1][1]
        for name, d in DIRECTIONS.items():
            if d == (dx, dy):
                return name
    return "up"


def simulate_turn(state: State, moves: dict) -> State:
    """Aplica um turno completo (todas as cobras ao mesmo tempo) seguindo as regras
    padrão do Battlesnake. Devolve um NOVO estado; o original não é alterado.

    Ordem: mover -> perder 1 de vida -> dano de hazard -> comer -> eliminações.
    """
    board = state.board
    w, h = board.width, board.height

    # 1) mover (cabeça nova, cauda sai) e 2) perder 1 de vida
    entries = []
    for s in state.snakes:
        mv = moves.get(s.id) or _keep_direction(s)
        dx, dy = DIRECTIONS[mv]
        hx, hy = s.body[0]
        entries.append([s.id, ((hx + dx, hy + dy),) + s.body[:-1], s.health - 1])

    # 3) dano de hazard
    if board.hazards:
        for e in entries:
            if e[1][0] in board.hazards:
                e[2] -= board.hazard_damage

    # 4) comer: vida volta a 100 e a cobra cresce (cauda duplicada)
    eaten = set()
    if board.food:
        for e in entries:
            head = e[1][0]
            if head in board.food:
                e[2] = 100
                e[1] = e[1] + (e[1][-1],)
                eaten.add(head)

    # 5) eliminações (simultâneas)
    body_cells: set = set()
    head_lengths: dict = {}
    for e in entries:
        body_cells.update(e[1][1:])
        head_lengths.setdefault(e[1][0], []).append(len(e[1]))

    survivors = []
    for sid, body, health in entries:
        hx, hy = body[0]
        if health <= 0 or not (0 <= hx < w and 0 <= hy < h) or body[0] in body_cells:
            continue
        lens = head_lengths[body[0]]
        if len(lens) > 1:                     # cabeça contra cabeça
            others = list(lens)
            others.remove(len(body))
            if any(l >= len(body) for l in others):   # perde se for menor ou igual
                continue
        survivors.append(Snake(sid, body, health))

    new_board = board
    if eaten:
        new_board = Board(w, h, board.food - eaten, board.hazards, board.hazard_damage)
    return State(new_board, survivors, state.me_id, state.turn + 1)


# =============================================================================
# 4. Segurança: filtros de morte certa
# =============================================================================

def head_danger_cells(state: State, snake: Snake) -> set:
    """Casas onde uma cobra de tamanho >= ao da `snake` pode chegar no próximo turno
    (um duelo de cabeças ali seria perdido ou empatado)."""
    table = neighbor_table(state.board.width, state.board.height)
    cells: set = set()
    for other in state.snakes:
        if other.id != snake.id and other.length >= snake.length:
            cells.update(table[other.head])
    return cells


def classify_moves(state: State, snake: Snake, blocked: set | None = None):
    """Devolve (seguras, arriscadas, fatais).

    fatal    : parede, corpo de qualquer cobra, ou morrer de fome no ato
    arriscada: pode haver duelo de cabeças com cobra igual/maior
    segura   : o resto
    """
    if blocked is None:
        blocked = blocked_cells(state)
    board = state.board
    danger = head_danger_cells(state, snake)
    hx, hy = snake.head
    safe, risky, fatal = [], [], []
    for mv, (dx, dy) in DIRECTIONS.items():
        target = (hx + dx, hy + dy)
        if not in_bounds(board, target) or target in blocked:
            fatal.append(mv)
        elif snake.health <= 1 and target not in board.food:
            fatal.append(mv)
        elif target in danger:
            risky.append(mv)
        else:
            safe.append(mv)
    return safe, risky, fatal


def non_fatal_moves(state: State, snake: Snake, blocked: set | None = None) -> list:
    safe, risky, _ = classify_moves(state, snake, blocked)
    return safe + risky


# =============================================================================
# 5a. Espaço livre (flood fill)
# =============================================================================

def flood_fill_count(board: Board, start: Point, blocked: set, limit: int | None = None) -> int:
    """Quantas casas livres dá para alcançar a partir de `start` (sem contar `start`).
    Com `limit`, para assim que atinge esse número (basta saber se "cabe")."""
    table = neighbor_table(board.width, board.height)
    seen = {start}
    stack = [start]
    count = 0
    while stack:
        cur = stack.pop()
        for nb in table[cur]:
            if nb in seen or nb in blocked:
                continue
            seen.add(nb)
            count += 1
            if limit is not None and count >= limit:
                return count
            stack.append(nb)
    return count


def space_after_move(state: State, snake: Snake, move: str,
                     blocked: set | None = None, limit: int | None = None) -> int:
    """Espaço livre que sobra depois de `snake` fazer `move` (conta a casa de destino).
    Devolve 0 se o movimento sai do tabuleiro ou bate em um corpo."""
    if blocked is None:
        blocked = blocked_cells(state)
    target = step(snake.head, move)
    if not in_bounds(state.board, target) or target in blocked:
        return 0
    return 1 + flood_fill_count(state.board, target, blocked, limit)


# =============================================================================
# 5b. Caminhos (BFS / A*)
# =============================================================================

@dataclass(slots=True)
class PathResult:
    move: str          # primeira jogada do caminho
    distance: int      # número de passos até o objetivo
    path: list         # casas do caminho (sem a casa inicial)


def move_toward(start: Point, cell: Point) -> str:
    """Direção que leva de `start` até a casa vizinha `cell`."""
    d = (cell[0] - start[0], cell[1] - start[1])
    for name, vec in DIRECTIONS.items():
        if vec == d:
            return name
    raise ValueError(f"{cell} não é vizinha de {start}")


def bfs_nearest(board: Board, start: Point, goals, blocked: set) -> list | None:
    """BFS: caminho mais curto até o objetivo mais próximo. None se inalcançável."""
    if not goals:
        return None
    table = neighbor_table(board.width, board.height)
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur != start and cur in goals:
            path = []
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nb in table[cur]:
            if nb in parent or (nb in blocked and nb not in goals):
                continue
            parent[nb] = cur
            queue.append(nb)
    return None


def astar(board: Board, start: Point, goal: Point, blocked: set) -> list | None:
    """A* (heurística de Manhattan) de `start` até `goal`. A casa `goal` é sempre
    considerada livre. Devolve as casas do caminho (sem `start`) ou None."""
    table = neighbor_table(board.width, board.height)
    open_heap = [(manhattan(start, goal), 0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        if g > g_cost.get(cur, 1 << 30):
            continue
        for nb in table[cur]:
            if nb in blocked and nb != goal:
                continue
            ng = g + 1
            if ng < g_cost.get(nb, 1 << 30):
                g_cost[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + manhattan(nb, goal), ng, nb))
    return None


def path_to_nearest_food(state: State, snake: Snake, blocked: set | None = None) -> PathResult | None:
    """Caminho até a comida mais próxima, evitando (se possível) casas de duelo perdido."""
    if blocked is None:
        blocked = blocked_cells(state)
    food = state.board.food
    path = bfs_nearest(state.board, snake.head, food, blocked | head_danger_cells(state, snake))
    if path is None:
        path = bfs_nearest(state.board, snake.head, food, blocked)
    if not path:
        return None
    return PathResult(move_toward(snake.head, path[0]), len(path), path)


def path_to_tail(state: State, snake: Snake, blocked: set | None = None) -> PathResult | None:
    """Caminho até a própria cauda (seguir a cauda é uma forma segura de ganhar tempo).
    Sem caminho se a cobra acabou de comer (a cauda fica parada um turno)."""
    if snake.length < 2 or just_ate(snake):
        return None
    if blocked is None:
        blocked = blocked_cells(state)
    path = astar(state.board, snake.head, snake.body[-1], blocked)
    if not path:
        return None
    return PathResult(move_toward(snake.head, path[0]), len(path), path)


# =============================================================================
# 6. Território (Voronoi)
# =============================================================================

@dataclass(slots=True)
class Territory:
    cells: dict   # id da cobra -> nº de casas que ela alcança primeiro
    food: dict    # id da cobra -> nº de comidas dentro do território dela


def compute_territory(state: State, blocked: set | None = None) -> Territory:
    """Voronoi eficiente para o duelo: menor distância vence; empate fica neutro."""
    board = state.board
    table = neighbor_table(board.width, board.height)
    if blocked is None:
        blocked = blocked_cells(state)

    snakes = state.snakes
    if not snakes:
        return Territory({}, {})
    if len(snakes) == 1:
        only = snakes[0]
        cells = flood_fill_count(board, only.head, blocked, None)
        food_count = sum(1 for f in board.food if f not in blocked)
        return Territory({only.id: cells}, {only.id: food_count})

    starts = [snakes[0].head, snakes[1].head]

    def distances(start: Point) -> dict[Point, int]:
        q = deque([start])
        dist = {start: 0}
        while q:
            cur = q.popleft()
            nd = dist[cur] + 1
            for nb in table[cur]:
                if nb in blocked or nb in dist:
                    continue
                dist[nb] = nd
                q.append(nb)
        return dist

    d0 = distances(starts[0])
    d1 = distances(starts[1])
    cells = [0, 0]
    food = [0, 0]

    for x in range(board.width):
        for y in range(board.height):
            cell = (x, y)
            if cell in blocked:
                continue
            a = d0.get(cell)
            b = d1.get(cell)
            if a is None and b is None:
                continue
            if a is not None and (b is None or a < b):
                cells[0] += 1
                if cell in board.food:
                    food[0] += 1
            elif b is not None and (a is None or b < a):
                cells[1] += 1
                if cell in board.food:
                    food[1] += 1

    return Territory(
        {snakes[0].id: cells[0], snakes[1].id: cells[1]},
        {snakes[0].id: food[0], snakes[1].id: food[1]},
    )


# =============================================================================
# 7. Avaliação do estado
# =============================================================================

def _legal_moves_for_state(state: State, snake: Snake) -> list[str]:
    blocked = blocked_cells(state)
    safe, risky, fatal = classify_moves(state, snake, blocked)
    return safe + risky


def _mobility_metrics(state: State, snake: Snake) -> tuple[int, int, int]:
    """(movimentos legais, saídas robustas, espaço acessível limitado)."""
    moves = _legal_moves_for_state(state, snake)
    robust = 0
    max_area = 0
    blocked = blocked_cells(state)
    for mv in moves:
        area = space_after_move(state, snake, mv, blocked, limit=snake.length + 12)
        max_area = max(max_area, area)
        if area >= snake.length + 2:
            robust += 1
    return len(moves), robust, max_area


def _tail_access_score(state: State, snake: Snake) -> float:
    """Valoriza uma cauda alcançável hoje OU depois que o corpo libera casas."""
    if snake.length < 2 or just_ate(snake):
        return 0.0

    blocked = blocked_cells(state)
    static_path = astar(state.board, snake.head, snake.body[-1], blocked)
    static_distance = None if static_path is None else len(static_path)
    temporal_distance = temporal_tail_distance(state, snake)
    free = flood_fill_count(state.board, snake.head, blocked, snake.length + 8)

    distance = temporal_distance if temporal_distance is not None else static_distance
    if distance is None:
        if free < snake.length + 3:
            return -1.5
        return 0.0

    score = 1.0 / (1.0 + distance)
    if distance <= 2:
        score += 0.45
    elif distance <= 4:
        score += 0.20

    # Vitória/sobrevivência pode depender de esperar o corpo abrir um corredor.
    if static_distance is None and temporal_distance is not None:
        score += 0.16

    return score


def _tail_distance(state: State, snake: Snake) -> int | None:
    """Distância até a cauda, preferindo a leitura temporal da abertura do corpo."""
    if snake.length < 2 or just_ate(snake):
        return None
    temporal = temporal_tail_distance(state, snake)
    if temporal is not None:
        return temporal
    path = astar(state.board, snake.head, snake.body[-1], blocked_cells(state))
    return None if path is None else len(path)


def _food_race_value(state: State, me: Snake, enemy: Snake) -> float:
    """Compara quem chega primeiro à comida considerando liberação das caudas."""
    if not state.board.food:
        return 0.0

    my_arrival = temporal_bfs(
        state, me.head, 0, me.length, state.board.width * state.board.height
    )
    enemy_arrival = temporal_bfs(
        state, enemy.head, 0, enemy.length, state.board.width * state.board.height
    )
    my_times = [my_arrival[f] for f in state.board.food if f in my_arrival]
    enemy_times = [enemy_arrival[f] for f in state.board.food if f in enemy_arrival]

    dm = min(my_times) if my_times else None
    de = min(enemy_times) if enemy_times else None
    if dm is None and de is None:
        return 0.0
    if dm is not None and de is None:
        return 1.5
    if dm is None:
        return -1.5

    advantage = de - dm
    health_factor = 1.0 if me.health <= 45 else 0.55
    return max(-4.0, min(4.0, advantage)) * health_factor


def _health_advantage(me: Snake, enemy: Snake) -> float:
    """Saúde restante como reserva de turnos, normalizada para não dominar a busca."""
    diff = max(-30, min(30, me.health - enemy.health))
    return diff / 30.0


def _enemy_cut_score(state: State, me: Snake, enemy: Snake) -> float:
    """Mede quão perto o rival está de uma posição sem saída.

    A ideia não é premiar simplesmente proximidade: premia quando a cobra adversária
    fica com poucas opções enquanto nós mantemos espaço suficiente.
    """
    enemy_moves, enemy_robust, enemy_area = _mobility_metrics(state, enemy)
    my_moves, my_robust, my_area = _mobility_metrics(state, me)
    score = 0.0
    if enemy_moves <= 1:
        score += 3.0
    elif enemy_moves == 2:
        score += 1.5
    if enemy_robust == 0:
        score += 2.5
    elif enemy_robust == 1:
        score += 1.0
    if enemy_area < enemy.length + 2:
        score += 4.0
    if my_robust >= 2 and my_area >= me.length + 5:
        score += 1.5
    return score


def _temporal_space_score(state: State, me: Snake, enemy: Snake) -> float:
    """Compara espaço atual com o espaço que se abre quando os corpos avançam."""
    board_cells = state.board.width * state.board.height
    _, _, my_area = _mobility_metrics(state, me)
    _, _, enemy_area = _mobility_metrics(state, enemy)
    trigger = (
        min(my_area, enemy_area) < max(me.length, enemy.length) + 10
        or max(me.length, enemy.length) >= 8
    )
    if not trigger:
        return 0.0

    limit = min(board_cells, max(me.length, enemy.length) + 22)
    my_arrival = temporal_bfs(state, me.head, 0, me.length, limit)
    enemy_arrival = temporal_bfs(state, enemy.head, 0, enemy.length, limit)
    my_temporal = len(my_arrival)
    enemy_temporal = len(enemy_arrival)

    score = W_TEMPORAL_SPACE_DIFF * (my_temporal - enemy_temporal)
    my_open = max(0, my_temporal - my_area)
    enemy_open = max(0, enemy_temporal - enemy_area)
    score += W_TEMPORAL_SPACE * min(10, my_open)
    score -= W_TEMPORAL_SPACE * min(10, enemy_open) * 0.7

    temporal_tail = temporal_tail_distance(state, me)
    static_path = None
    if me.length >= 2 and not just_ate(me):
        static_path = astar(state.board, me.head, me.body[-1], blocked_cells(state))
    if temporal_tail is not None and static_path is None:
        score += W_TEMPORAL_TAIL
    elif temporal_tail is None and my_area < me.length + 6:
        score -= W_TEMPORAL_TAIL * 0.5

    if my_open >= 4 and my_area < me.length + 8:
        score += W_CORRIDOR_OPENING
    return score


def evaluate(state: State) -> float:
    """Avaliação v7 para 1v1: espaço + cerco + cauda + comida + saúde.

    A ideia central é diferenciar uma vantagem territorial comum de uma posição
    que pode realmente virar vitória forçada.
    """
    me = state.me()
    enemies = state.enemies()

    if me is None:
        return DRAW_SCORE if not state.snakes else -WIN_SCORE
    if not enemies:
        return WIN_SCORE

    enemy = enemies[0]
    board = state.board
    blocked = blocked_cells(state)
    score = 0.0

    # 1) Território relativo. Empates ficam neutros no Voronoi; tamanho é tratado
    # separadamente por causa do head-to-head.
    terr = compute_territory(state, blocked)
    my_cells = terr.cells.get(me.id, 0)
    enemy_cells = terr.cells.get(enemy.id, 0)
    territory_diff = my_cells - enemy_cells
    score += W_TERRITORY * territory_diff

    # 2) Espaço e mobilidade.
    my_moves, my_robust, my_area = _mobility_metrics(state, me)
    enemy_moves, enemy_robust, enemy_area = _mobility_metrics(state, enemy)
    score += W_MOBILITY * my_moves
    score -= W_ENEMY_MOBILITY * enemy_moves
    score += W_ROBUST_EXITS * my_robust
    score -= W_ENEMY_ROBUST_EXITS * enemy_robust
    score += 0.40 * (my_area - enemy_area)

    # 2b) Espaço temporal: uma posição pode parecer fechada agora, mas ficar aberta
    # quando a cauda/corpos avançarem.
    score += _temporal_space_score(state, me, enemy)

    # 3) Corte territorial: queremos reduzir o espaço do rival sem nos fechar.
    cut = _enemy_cut_score(state, me, enemy)
    score += W_ATTACK_PRESSURE * cut
    if enemy_area < enemy.length:
        score += W_ENEMY_TRAPPED * (enemy.length - enemy_area)

    # Fica ainda mais valioso se já tivermos pelo menos duas saídas robustas.
    if enemy_moves <= 2 and my_robust >= 2:
        score += W_FORCING * (3 - enemy_moves)

    if enemy_moves <= 1 and my_area >= me.length + 4:
        score += W_FORCING * 1.35

    # 4) Cauda: em posição fechada ela vale quase tanto quanto comida.
    my_tail = _tail_access_score(state, me)
    enemy_tail = _tail_access_score(state, enemy)
    score += W_TAIL * my_tail
    score -= 0.55 * W_TAIL * enemy_tail

    my_tail_dist = _tail_distance(state, me)
    if my_tail_dist is not None:
        score += W_TAIL_DISTANCE / (1.0 + my_tail_dist)
    else:
        if my_area < me.length + 5:
            score -= W_TAIL_DISTANCE * 0.35

    # 5) Tamanho: vantagem de cabeça, limitada.
    diff = me.length - enemy.length
    score += W_LENGTH * max(-LENGTH_CAP, min(LENGTH_CAP, diff))

    # 6) Saúde: útil especialmente quando a partida vira uma disputa de território.
    score += W_HEALTH * _health_advantage(me, enemy) * 30.0

    # 7) Comida: recupera o melhor da v5, mas sem permitir que comida substitua
    # uma posição taticamente perdida.
    if board.food:
        my_food_dist = min((manhattan(me.head, f) for f in board.food), default=None)
        if my_food_dist is not None:
            hunger = max(0.0, HUNGER_START - me.health) / HUNGER_START
            score -= W_HUNGER * hunger * my_food_dist
            if me.health <= my_food_dist:
                score -= STARVE_PENALTY

        score += W_FOOD_TERRITORY * terr.food.get(me.id, 0)
        score += W_FOOD_RACE * _food_race_value(state, me, enemy)

        # A comida próxima e exclusiva é melhor do que a mesma comida disputada.
        enemy_food_dist = min((manhattan(enemy.head, f) for f in board.food), default=None)
        if enemy_food_dist is not None and my_food_dist is not None:
            if my_food_dist + 1 < enemy_food_dist:
                score += 4.0
            elif enemy_food_dist + 1 < my_food_dist:
                score -= 4.0

    # 8) Centro quase não importa em comparação com controle de espaço.
    cx, cy = (board.width - 1) / 2, (board.height - 1) / 2
    score -= W_CENTER * (abs(me.head[0] - cx) + abs(me.head[1] - cy))

    # 9) Final de jogo: sem comida ou com cobras grandes, corte/cauda/saídas ganham
    # importância porque normalmente a partida é decidida pelo espaço restante.
    if not board.food or min(me.length, enemy.length) >= 12:
        score += W_ENDGAME_CUT * cut
        score += 0.8 * W_TAIL_DISTANCE / (1.0 + (my_tail_dist or 99))

    # 10) Tensão de finalização: quando somos maiores e já limitamos o rival,
    # aproximar a cabeça é útil; entrar numa casa de head-to-head perdido continua proibido.
    if me.length > enemy.length:
        distance = manhattan(me.head, enemy.head)
        if enemy_moves <= 2:
            score += W_WINNING_TENSION * max(0, 5 - distance)

    return score


# =============================================================================
# 7b. Detecção de vitória forçada
# =============================================================================

def _enemy_moves_after_my_move(state: State, my_move: str) -> tuple[State, list[str]] | None:
    """Simula nossa jogada e retorna o novo estado + respostas legais do rival."""
    me = state.me()
    if me is None:
        return None
    enemies = state.enemies()
    if not enemies:
        return None
    enemy = enemies[0]
    child = simulate_turn(state, {me.id: my_move, enemy.id: _keep_direction(enemy)})
    alive_me = child.me()
    if alive_me is None:
        return None
    child_enemy = child.enemies()
    if not child_enemy:
        return child, []
    return child, _ordered_moves(child, child_enemy[0], blocked_cells(child))


def _wins_after_all_enemy_replies(state: State, my_move: str) -> bool:
    """Verdadeiro quando uma jogada mata/encurrala o rival independentemente da resposta."""
    me = state.me()
    enemies = state.enemies()
    if me is None or not enemies:
        return False
    enemy = enemies[0]
    blocked = blocked_cells(state)
    enemy_moves = _ordered_moves(state, enemy, blocked)

    # Sem jogadas não fatais: a cobra rival já está em zugzwang/mate.
    if not enemy_moves:
        child = simulate_turn(state, {me.id: my_move, enemy.id: _keep_direction(enemy)})
        return child.me() is not None and not child.enemies()

    for em in enemy_moves:
        child = simulate_turn(state, {me.id: my_move, enemy.id: em})
        if child.me() is None:
            return False
        if child.enemies():
            return False
    return True


def _find_forced_win(state: State, candidates: list[str], deadline: float) -> tuple[str | None, int]:
    """Procura mate em 1 ou mate forçado em 2 antes da busca normal.

    Retorna (movimento, nós). A segunda camada só é usada quando nenhum mate em 1
    foi encontrado. Isso evita deixar uma vitória forçada escapar por uma heurística.
    """
    if not state.enemies() or not candidates:
        return None, 0

    nodes = 0

    # Primeiro: mate em 1.
    for mv in candidates:
        if time.perf_counter() > deadline:
            return None, nodes
        nodes += 1
        if _wins_after_all_enemy_replies(state, mv):
            return mv, nodes

    # Segundo: para cada resposta rival, existe uma nossa segunda jogada vencedora.
    me = state.me()
    enemy = state.enemies()[0]
    enemy_moves = _ordered_moves(state, enemy, blocked_cells(state))
    if not enemy_moves:
        return None, nodes

    for first_mv in candidates:
        if time.perf_counter() > deadline:
            return None, nodes
        forced_for_every_reply = True

        for enemy_mv in enemy_moves:
            if time.perf_counter() > deadline:
                return None, nodes
            nodes += 1
            child = simulate_turn(state, {me.id: first_mv, enemy.id: enemy_mv})

            if child.me() is None:
                forced_for_every_reply = False
                break
            if not child.enemies():
                continue

            child_me = child.me()
            child_blocked = blocked_cells(child)
            child_candidates = _ordered_moves(child, child_me, child_blocked)
            if not child_candidates:
                forced_for_every_reply = False
                break

            found_second = False
            for second_mv in child_candidates:
                if time.perf_counter() > deadline:
                    return None, nodes
                nodes += 1
                if _wins_after_all_enemy_replies(child, second_mv):
                    found_second = True
                    break

            if not found_second:
                forced_for_every_reply = False
                break

        if forced_for_every_reply:
            return first_mv, nodes

    return None, nodes


# =============================================================================
# 8. Busca (minimax / alpha-beta)
# =============================================================================

INF = float("inf")


class SearchTimeout(Exception):
    pass


@dataclass
class _Ctx:
    me_id: str
    deadline: float
    had_enemies: bool
    nodes: int = 0
    tt: dict = field(default_factory=dict)

    def tick(self) -> None:
        self.nodes += 1
        if time.perf_counter() > self.deadline:
            raise SearchTimeout


@dataclass(slots=True)
class SearchResult:
    scores: dict    # jogada -> nota da última profundidade completa
    depth: int      # profundidade completa alcançada (0 = nenhuma)
    nodes: int      # nós visitados


def _move_order_score(state: State, snake: Snake, mv: str, blocked: set) -> float:
    """Ordena movimentos pelo potencial tático antes da poda alpha-beta."""
    target = step(snake.head, mv)
    if target in blocked or not in_bounds(state.board, target):
        return -10_000.0

    area = space_after_move(state, snake, mv, blocked, limit=snake.length + 16)
    value = area * 1.7

    # Em corredores, prioriza também movimentos cujo espaço cresce com a liberação do corpo.
    if snake.length >= 6 or area < snake.length + 10:
        temporal_area, _ = temporal_space_after_move(
            state, snake, mv, limit=snake.length + 22
        )
        value += 1.15 * temporal_area
        if temporal_area > area + 3:
            value += W_CORRIDOR_OPENING

    enemies = state.enemies() if snake.id == state.me_id else [s for s in state.snakes if s.id != snake.id]
    if enemies:
        enemy = enemies[0]
        enemy_moves = len(_legal_moves_for_state(state, enemy))
        dist = manhattan(target, enemy.head)
        if snake.length > enemy.length:
            value += max(0, 6 - dist) * 6
            if enemy_moves <= 2:
                value += 24
        elif snake.length <= enemy.length and dist <= 2:
            value -= 30

        # Tenta ordenar primeiro movimentos que cortam o território do rival.
        child = simulate_turn(state, {snake.id: mv, enemy.id: _keep_direction(enemy)})
        child_enemy = child.enemies()
        if child.me() is not None and child_enemy:
            e = child_enemy[0]
            e_moves, e_robust, e_area = _mobility_metrics(child, e)
            value += max(0, enemy_moves - e_moves) * 8
            value += max(0, 2 - e_robust) * 5
            if e_area < e.length + 3:
                value += 14

    if target in state.board.food:
        value += 7 if snake.health < 55 else 2
    if target in state.board.hazards:
        value -= 30
    if snake.length >= 2 and not just_ate(snake):
        if target == snake.body[-1]:
            value += 8
    return value


def _ordered_moves(state: State, snake: Snake, blocked: set) -> list:
    """Jogadas legais, colocando primeiro as candidatas com maior potencial de poda."""
    safe, risky, fatal = classify_moves(state, snake, blocked)
    moves = safe + risky
    moves.sort(key=lambda mv: _move_order_score(state, snake, mv, blocked), reverse=True)
    return moves


def _enemy_combos(state: State, me_id: str, blocked: set) -> list:
    """Combinações de jogadas dos rivais (cada rival só joga o que não o mata na hora)."""
    me = state.me()
    enemies = [s for s in state.snakes if s.id != me_id]
    if not enemies:
        return [{}]
    enemies.sort(key=lambda e: manhattan(e.head, me.head))
    option_lists = []
    for idx, e in enumerate(enemies):
        moves = _ordered_moves(state, e, blocked) or ["up"]   # sem saída: vai morrer de qualquer jeito
        if idx >= BRANCHED_ENEMIES:
            moves = moves[:1]
        option_lists.append([(e.id, m) for m in moves])
    return [dict(c) for c in itertools.product(*option_lists)]


def _state_key(state: State, depth: int, side: str, my_move: str | None = None):
    """Chave compacta para transposição dentro de uma única busca."""
    snakes = tuple(sorted((s.id, s.health, s.body) for s in state.snakes))
    return (
        side,
        depth,
        my_move,
        state.me_id,
        state.board.width,
        state.board.height,
        tuple(sorted(state.board.food)),
        tuple(sorted(state.board.hazards)),
        state.board.hazard_damage,
        snakes,
    )


def _max_node(state: State, depth: int, alpha: float, beta: float, ctx: _Ctx) -> float:
    ctx.tick()
    me = state.me()
    if me is None:
        return DRAW_SCORE if not state.snakes else -(WIN_SCORE + depth)
    if ctx.had_enemies and len(state.snakes) == 1:
        return WIN_SCORE + depth
    if depth == 0:
        return evaluate(state)

    key = _state_key(state, depth, "MAX")
    cached = ctx.tt.get(key)
    if cached is not None:
        return cached

    blocked = blocked_cells(state)
    value = -INF
    cutoff = False
    moves = _ordered_moves(state, me, blocked) or ["up"]
    for mv in moves:
        v = _min_node(state, mv, depth, alpha, beta, ctx)
        if v > value:
            value = v
        if value > alpha:
            alpha = value
        if alpha >= beta:
            cutoff = True
            break

    # Só guardamos valores exatos. Nós cortados são limites e não podem ser reutilizados
    # como se fossem valores verdadeiros.
    if not cutoff:
        ctx.tt[key] = value
    return value


def _min_node(state: State, my_move: str, depth: int, alpha: float, beta: float, ctx: _Ctx) -> float:
    ctx.tick()
    key = _state_key(state, depth, "MIN", my_move)
    cached = ctx.tt.get(key)
    if cached is not None:
        return cached

    blocked = blocked_cells(state)
    value = INF
    cutoff = False
    for combo in _enemy_combos(state, ctx.me_id, blocked):
        moves = dict(combo)
        moves[ctx.me_id] = my_move
        child = simulate_turn(state, moves)
        v = _max_node(child, depth - 1, alpha, beta, ctx)
        if v < value:
            value = v
        if value < beta:
            beta = value
        if beta <= alpha:
            cutoff = True
            break

    if not cutoff:
        ctx.tt[key] = value
    return value


def _root_pass(state: State, depth: int, ctx: _Ctx, root_moves: list, first: str | None) -> dict:
    """Uma profundidade completa com valores exatos na raiz."""
    moves = list(root_moves)
    if first in moves:
        moves.remove(first)
        moves.insert(0, first)
    scores: dict = {}
    best = -INF
    for mv in moves:
        v = _min_node(state, mv, depth, -INF, INF, ctx)
        scores[mv] = v
        if v > best:
            best = v
    return scores


def search_root(state: State, deadline: float, root_moves: list) -> SearchResult:
    """Aprofundamento iterativo a partir do estado atual."""
    ctx = _Ctx(state.me_id, deadline, had_enemies=len(state.snakes) > 1)
    best_scores: dict = {}
    reached = 0
    first = None
    for depth in range(1, MAX_DEPTH + 1):
        try:
            scores = _root_pass(state, depth, ctx, root_moves, first)
        except SearchTimeout:
            break
        best_scores, reached = scores, depth
        first = max(scores, key=scores.get)
        if scores[first] >= WIN_SCORE - 1000:   # vitória forçada: não precisa mais buscar
            break
    return SearchResult(best_scores, reached, ctx.nodes)


# =============================================================================
# 9. Estratégia
# =============================================================================

@dataclass(slots=True)
class Decision:
    move: str
    reason: str
    depth: int = 0
    nodes: int = 0
    elapsed_ms: float = 0.0


def _last_resort(state: State, me: Snake) -> str:
    """Fallback determinístico: escolhe a saída com maior espaço."""
    blocked = blocked_cells(state)
    candidates = [m for m in DIRECTIONS if in_bounds(state.board, step(me.head, m)) and step(me.head, m) not in blocked]
    if not candidates:
        return "up"
    return max(candidates, key=lambda m: space_after_move(state, me, m, blocked, limit=me.length + 20))


def _heuristic_pick(state: State, me: Snake, candidates: list, blocked: set) -> str:
    """Plano B determinístico com foco em espaço + pressão."""
    return max(candidates, key=lambda mv: _move_order_score(state, me, mv, blocked))


def choose_move(state: State, budget_s: float = 0.25) -> Decision:
    t0 = time.perf_counter()
    me = state.me()
    if me is None:
        return Decision("up", "cobra não encontrada")

    blocked = blocked_cells(state)
    safe, risky, _ = classify_moves(state, me, blocked)
    candidates = safe + risky

    if not candidates:
        return Decision(_last_resort(state, me), "sem saída", elapsed_ms=(time.perf_counter() - t0) * 1000)
    if len(candidates) == 1:
        return Decision(candidates[0], "única jogada viável", elapsed_ms=(time.perf_counter() - t0) * 1000)

    deadline = t0 + max(MIN_BUDGET_S, budget_s)

    # Antes do minimax, procura vitórias táticas que podem ser provadas de forma
    # direta. Em 1v1, isso é especialmente valioso quando o rival está quase sem saídas.
    if FORCED_WIN_CHECK and state.enemies():
        tactical_deadline = min(
            deadline,
            t0 + max(0.008, budget_s * FORCED_WIN_BUDGET_FRACTION),
        )
        forced_move, tactical_nodes = _find_forced_win(state, candidates, tactical_deadline)
        if forced_move is not None:
            return Decision(
                forced_move,
                "vitória forçada (cerco)",
                depth=2,
                nodes=tactical_nodes,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    result = search_root(state, deadline, candidates)

    if not result.scores:
        move = _heuristic_pick(state, me, candidates, blocked)
        reason = "heurística (tempo esgotado)"
    else:
        move = max(result.scores, key=result.scores.get)
        reason = "minimax 1v1 v8 + espaço temporal + cerco"

    return Decision(move, reason, result.depth, result.nodes, (time.perf_counter() - t0) * 1000)


# =============================================================================
# 10. Funções chamadas pelo jogo
# =============================================================================

def info() -> dict:
    """GET / — chamado quando você cadastra a cobra e a cada partida.
    Controla a aparência dela.
    Opções de cabeça, cauda e cor: https://docs.battlesnake.com/guides/customizations
    """
    logger.info("INFO")

    return {
        "apiversion": "1",
        "author": "PedroAAmaral",  # seu usuário do Battlesnake
        "color": "#8B0000",
        "head": "tiger-king",
        "tail": "hook",
        "version": "8.0.0",
    }


def start(state: GameState) -> None:
    """POST /start — chamado uma vez, quando a partida começa."""
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)


def end(state: GameState) -> None:
    """POST /end — chamado uma vez, quando a partida termina."""
    logger.info("FIM DE JOGO após %d turnos", state.turn)


def _budget_seconds(state: GameState) -> float:
    """Tempo que a busca pode usar neste turno.

    Fração do timeout do jogo (o resto fica para rede, API Gateway e Lambda). Se o
    jogo informar que a jogada ANTERIOR demorou perto do limite (`you.latency`),
    a fração é reduzida para não perder a jogada por tempo.
    """
    try:
        timeout_ms = int(getattr(getattr(state, "game", None), "timeout", None) or 500)
    except (TypeError, ValueError):
        timeout_ms = 500
    fraction = TIME_BUDGET_FRACTION
    try:
        latency_ms = int(getattr(state.you, "latency", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        latency_ms = 0
    if latency_ms >= LATENCY_ALERT * timeout_ms:
        fraction *= LATENCY_REDUCTION
    return max(MIN_BUDGET_S, timeout_ms / 1000.0 * fraction)


def _simple_fallback(state: GameState) -> str:
    """Rede de segurança simples e determinística."""
    try:
        deltas = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        head = state.you.body[0]
        occupied = {(p.x, p.y) for s in state.board.snakes for p in s.body[:-1]}
        ok = []
        for name, (dx, dy) in deltas.items():
            x, y = head.x + dx, head.y + dy
            if 0 <= x < state.board.width and 0 <= y < state.board.height and (x, y) not in occupied:
                ok.append(name)
        if not ok:
            return "up"
        # Mais vizinhos livres = menor chance de entrar em corredor morto.
        def local_space(mv):
            x, y = head.x + deltas[mv][0], head.y + deltas[mv][1]
            return sum(
                1 for dx, dy in deltas.values()
                if 0 <= x + dx < state.board.width
                and 0 <= y + dy < state.board.height
                and (x + dx, y + dy) not in occupied
            )
        return max(ok, key=local_space)
    except Exception:
        return "up"


def get_move(state: GameState) -> MoveResponse:
    """POST /move — chamado a cada turno. Precisa devolver "up", "down", "left" ou "right"."""
    try:
        decision = choose_move(from_api(state), _budget_seconds(state))
        logger.info(
            "MOVE %d: %s (%s | profundidade=%d nós=%d %.0fms)",
            state.turn, decision.move, decision.reason,
            decision.depth, decision.nodes, decision.elapsed_ms,
        )
        return MoveResponse(move=decision.move)
    except Exception:
        logger.exception("MOVE %s: erro no motor, usando fallback simples", getattr(state, "turn", "?"))
        return MoveResponse(move=_simple_fallback(state))
