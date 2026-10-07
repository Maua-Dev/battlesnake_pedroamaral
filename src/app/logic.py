
from __future__ import annotations

import heapq
import itertools
import logging
import random
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
TIME_BUDGET_FRACTION = 0.4   # fração do timeout do jogo usada na busca (resto: rede + Lambda)
MIN_BUDGET_S = 0.05          # nunca busca por menos que isso
MAX_DEPTH = 12               # profundidade máxima (em turnos completos); o tempo costuma parar antes
BRANCHED_ENEMIES = 2         # só os N rivais mais próximos ramificam; os demais seguem jogada padrão

# ----------------------------- Regras ----------------------------------------
DEFAULT_HAZARD_DAMAGE = 14   # dano extra por turno em zona de hazard (royale)

# ----------------------------- Notas terminais -------------------------------
WIN_SCORE = 1_000_000.0      # vitória (somada à profundidade restante: vitória rápida vale mais)
DRAW_SCORE = -100_000.0      # todos morrem juntos: melhor que perder, pior que viver

# ----------------------------- Pesos da avaliação ----------------------------
W_TERRITORY = 1.0            # por casa de vantagem de território (Voronoi)
W_LENGTH = 5.0               # por segmento de vantagem de tamanho (limitado a ±LENGTH_CAP)
LENGTH_CAP = 4
W_FOOD_TERRITORY = 2.0       # por comida dentro do meu território
W_HUNGER = 8.0               # penalidade: fome (0..1) x distância até a comida mais próxima
HUNGER_START = 60            # abaixo dessa vida a cobra começa a se preocupar com comida
STARVE_PENALTY = 400.0       # vida menor que a distância até qualquer comida
W_TRAPPED = 40.0             # por casa que falta para eu caber no espaço onde estou
W_ENEMY_TRAPPED = 15.0       # por casa que falta para um rival caber (cerco!)
TRAPPED_MARGIN = 6           # limite do flood fill = tamanho + margem
W_CENTER = 0.8               # por casa de distância do centro

# ----------------------------- Orientação da estratégia ----------------------
GUIDANCE_TOLERANCE = 15.0    # a jogada guiada só vale se a nota ficar dentro dessa margem da melhor
HUNGER_GUIDANCE_HEALTH = 35  # com vida <= isso, prefere o caminho (BFS) até a comida
CRAMPED_FACTOR = 1.5         # espaço < fator x tamanho => "apertado": segue a própria cauda

# ----------------------------- Território e cerco ----------------------------
TEMPORAL_TERRITORY = True    # território considera caudas/corpos que liberam casas com o tempo
TEMPORAL_ESCAPE = True       # detecta "preso" também levando em conta corpos que liberam casas
INTERIOR_BONUS = 0.5         # casa do centro vale 1 + bônus (casa da borda vale 1): rouba o centro
W_EDGE_ENEMY = 4.0           # bônus: rival colado na parede e eu domino mais espaço
W_CORNER_ENEMY = 3.0         # bônus: rival perto de um canto e eu domino mais espaço

LATENCY_ALERT = 0.8        # se a jogada anterior levou >= 80% do timeout...
LATENCY_REDUCTION = 0.6    # ...a busca usa só 60% do orçamento normal


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
    _release: dict | None = field(default=None, repr=False, compare=False)  # cache

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
    cells: dict    # id da cobra -> nº de casas que ela alcança primeiro
    wcells: dict   # id da cobra -> mesmas casas, ponderadas (centro vale mais que a borda)
    food: dict     # id da cobra -> nº de comidas dentro do território dela


@lru_cache(maxsize=8)
def _cell_weights(width: int, height: int) -> dict:
    """Peso de cada casa no território: borda = 1; centro = 1 + INTERIOR_BONUS.
    Com INTERIOR_BONUS = 0 todas valem 1. Roubar o centro do rival empurra ele para
    as paredes, onde as saídas são poucas."""
    weights = {}
    for x in range(width):
        for y in range(height):
            d = min(x, y, width - 1 - x, height - 1 - y)
            weights[(x, y)] = 1.0 + INTERIOR_BONUS * min(d, 2) / 2.0
    return weights


def release_times(state: State) -> dict:
    """Casa -> em quantos turnos ela fica livre. Um segmento a `i` posições da cabeça
    sai do lugar daqui a (tamanho - i) turnos: a cauda libera em 1, o pescoço só no fim.
    Corpo empilhado (cobra que acabou de comer) conta pela maior espera."""
    cached = state._release
    if cached is not None:
        return cached
    rel: dict = {}
    for s in state.snakes:
        n = len(s.body)
        for i, cell in enumerate(s.body):
            r = n - i
            if r > rel.get(cell, 0):
                rel[cell] = r
    state._release = rel
    return rel


def temporal_reach_count(state: State, start: Point, limit: int) -> int:
    """Quantas casas dá para alcançar a partir de `start` chegando nelas DEPOIS que o
    corpo que as ocupa já saiu (cobra enrolada: seguir a própria cauda). Para em `limit`."""
    table = neighbor_table(state.board.width, state.board.height)
    rel = release_times(state)
    seen = {start}
    frontier = [start]
    count = 0
    t = 0
    while frontier:
        t += 1
        nxt = []
        for cur in frontier:
            for nb in table[cur]:
                if nb in seen or rel.get(nb, 0) > t:
                    continue
                seen.add(nb)
                nxt.append(nb)
                count += 1
                if count >= limit:
                    return count
        frontier = nxt
    return count


def compute_territory(state: State, blocked: set | None = None) -> Territory:
    """Todas as cobras se expandem ao mesmo tempo, uma casa por vez.
    Empate de distância: leva a cobra MAIOR; se o tamanho também empata, ninguém leva
    (a casa fica disputada e ninguém passa por ela).

    Com TEMPORAL_TERRITORY, uma casa ocupada por corpo só é bloqueada até o turno em que
    o corpo sai dela (a cauda libera no turno 1). Sem ele, todo corpo é parede eterna."""
    board = state.board
    table = neighbor_table(board.width, board.height)
    weights = _cell_weights(board.width, board.height)
    temporal = TEMPORAL_TERRITORY
    rel = release_times(state) if temporal else None
    if blocked is None:
        blocked = blocked_cells(state)
    snakes = state.snakes
    n = len(snakes)
    lengths = [s.length for s in snakes]
    foods = board.food

    owner: dict = {}
    frontiers = []
    cells = [0] * n
    wcells = [0.0] * n
    food = [0] * n
    for i, s in enumerate(snakes):
        owner[s.head] = i
        frontiers.append([s.head])

    t = 0
    while True:
        t += 1
        claims: dict = {}
        for i, frontier in enumerate(frontiers):
            for cur in frontier:
                for nb in table[cur]:
                    if nb in owner:
                        continue
                    if temporal:
                        if rel.get(nb, 0) > t:
                            continue
                    elif nb in blocked:
                        continue
                    lst = claims.get(nb)
                    if lst is None:
                        claims[nb] = [i]
                    elif i not in lst:
                        lst.append(i)
        if not claims:
            break
        frontiers = [[] for _ in range(n)]
        for cell, ids in claims.items():
            if len(ids) == 1:
                winner = ids[0]
            else:
                best = max(lengths[i] for i in ids)
                top = [i for i in ids if lengths[i] == best]
                if len(top) > 1:
                    owner[cell] = -1
                    continue
                winner = top[0]
            owner[cell] = winner
            cells[winner] += 1
            wcells[winner] += weights[cell]
            frontiers[winner].append(cell)
            if cell in foods:
                food[winner] += 1

    return Territory(
        {snakes[i].id: cells[i] for i in range(n)},
        {snakes[i].id: wcells[i] for i in range(n)},
        {snakes[i].id: food[i] for i in range(n)},
    )


# =============================================================================
# 7. Avaliação do estado
# =============================================================================

def _enemy_squeeze(head: Point, board: Board) -> float:
    """Bônus por rival com pouca saída: junto de uma parede e/ou perto de um canto."""
    x, y = head
    last_x, last_y = board.width - 1, board.height - 1
    bonus = 0.0
    d_wall = min(x, y, last_x - x, last_y - y)
    if d_wall <= 1:
        bonus += W_EDGE_ENEMY * (2 - d_wall)
    d_corner = min(abs(x - cx) + abs(y - cy) for cx in (0, last_x) for cy in (0, last_y))
    if d_corner <= 3:
        bonus += W_CORNER_ENEMY * (4 - d_corner)
    return bonus


def evaluate(state: State) -> float:
    me = state.me()
    if me is None:
        return -WIN_SCORE
    enemies = state.enemies()
    board = state.board
    blocked = blocked_cells(state)
    terr = compute_territory(state, blocked)
    score = 0.0

    # 1) Território (Voronoi): minhas casas contra a média dos rivais
    my_cells = terr.cells[me.id]
    if enemies:
        enemy_avg = sum(terr.cells[e.id] for e in enemies) / len(enemies)
        enemy_w = sum(terr.wcells[e.id] for e in enemies) / len(enemies)
        score += W_TERRITORY * (terr.wcells[me.id] - enemy_w)
        # 2) Tamanho: cobra maior vence duelos de cabeça
        diff = me.length - max(e.length for e in enemies)
        score += W_LENGTH * max(-LENGTH_CAP, min(LENGTH_CAP, diff))
    else:
        score += W_TERRITORY * my_cells

    # 3) Comida no meu território
    score += W_FOOD_TERRITORY * terr.food[me.id]

    # 4) Fome: perto de morrer de fome, ficar longe da comida é ruim
    if board.food:
        dist = min(manhattan(me.head, f) for f in board.food)
        if me.health < dist:
            score -= STARVE_PENALTY
        hunger = max(0.0, HUNGER_START - me.health) / HUNGER_START
        score -= W_HUNGER * hunger * dist

    # 5) Espaço: preso numa área menor que o meu corpo é praticamente morte
    free = flood_fill_count(board, me.head, blocked, me.length + TRAPPED_MARGIN)
    if TEMPORAL_ESCAPE and free < me.length + TRAPPED_MARGIN:
        # só quando o espaço aperta: conta também as casas que o meu corpo vai liberar
        free = max(free, temporal_reach_count(state, me.head, me.length + TRAPPED_MARGIN))
    if free < me.length:
        score -= W_TRAPPED * (me.length - free)

    # 6) Cerco: rival preso numa área menor que o corpo dele é bom para mim
    for e in enemies:
        e_free = flood_fill_count(board, e.head, blocked, e.length + TRAPPED_MARGIN)
        if TEMPORAL_ESCAPE and e_free < e.length + TRAPPED_MARGIN:
            e_free = max(e_free, temporal_reach_count(state, e.head, e.length + TRAPPED_MARGIN))
        if e_free < e.length:
            score += W_ENEMY_TRAPPED * (e.length - e_free)

    # 6b) Cerco em bordas e cantos: rival colado na parede ou perto de um canto, e eu
    #     dominando mais espaço que ele (o espaço dele só tende a diminuir).
    if enemies and my_cells > enemy_avg and (W_EDGE_ENEMY or W_CORNER_ENEMY):
        for e in enemies:
            score += _enemy_squeeze(e.head, board)

    # 7) Centro: evita bordas e cantos
    cx, cy = (board.width - 1) / 2, (board.height - 1) / 2
    score -= W_CENTER * (abs(me.head[0] - cx) + abs(me.head[1] - cy))
    return score


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

    def tick(self) -> None:
        self.nodes += 1
        if time.perf_counter() > self.deadline:
            raise SearchTimeout


@dataclass(slots=True)
class SearchResult:
    scores: dict    # jogada -> nota da última profundidade completa
    depth: int      # profundidade completa alcançada (0 = nenhuma)
    nodes: int      # nós visitados


def _ordered_moves(state: State, snake: Snake, blocked: set) -> list:
    """Jogadas não fatais, melhores primeiro (seguras antes das arriscadas; mais casas
    livres ao redor primeiro). Boa ordenação = mais cortes na poda alpha-beta."""
    safe, risky, _ = classify_moves(state, snake, blocked)
    table = neighbor_table(state.board.width, state.board.height)
    hx, hy = snake.head
    ranked = []
    for tier, group in ((0, safe), (1, risky)):
        for mv in group:
            dx, dy = DIRECTIONS[mv]
            free = sum(1 for nb in table[(hx + dx, hy + dy)] if nb not in blocked)
            ranked.append((tier, -free, mv))
    ranked.sort()
    return [mv for _, _, mv in ranked]


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


def _max_node(state: State, depth: int, alpha: float, beta: float, ctx: _Ctx) -> float:
    ctx.tick()
    me = state.me()
    if me is None:   # eu morri
        return DRAW_SCORE if not state.snakes else -(WIN_SCORE + depth)
    if ctx.had_enemies and len(state.snakes) == 1:   # só sobrei eu
        return WIN_SCORE + depth
    if depth == 0:
        return evaluate(state)

    blocked = blocked_cells(state)
    value = -INF
    for mv in _ordered_moves(state, me, blocked) or ["up"]:
        v = _min_node(state, mv, depth, alpha, beta, ctx)
        if v > value:
            value = v
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break
    return value


def _min_node(state: State, my_move: str, depth: int, alpha: float, beta: float, ctx: _Ctx) -> float:
    blocked = blocked_cells(state)
    value = INF
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
            break
    return value


def _root_pass(state: State, depth: int, ctx: _Ctx, root_moves: list, first: str | None) -> dict:
    """Uma busca completa de profundidade `depth`. A janela do alpha usa a tolerância
    da estratégia, então jogadas dentro da tolerância recebem nota EXATA e as piores
    recebem apenas um limite superior (já fora da tolerância)."""
    moves = list(root_moves)
    if first in moves:
        moves.remove(first)
        moves.insert(0, first)
    scores: dict = {}
    best = -INF
    for mv in moves:
        alpha = best - GUIDANCE_TOLERANCE if best > -INF else -INF
        v = _min_node(state, mv, depth, alpha, INF, ctx)
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
    """Todas as jogadas matam: pelo menos fica no tabuleiro e não volta pelo pescoço."""
    neck = me.body[1] if me.length >= 2 else None
    options = [m for m in DIRECTIONS
               if in_bounds(state.board, step(me.head, m)) and step(me.head, m) != neck]
    return random.choice(options or list(DIRECTIONS))


def _heuristic_pick(state: State, me: Snake, candidates: list, blocked: set) -> str:
    """Plano B sem busca: mais espaço livre; desempata indo em direção à comida."""
    food_move = None
    res = path_to_nearest_food(state, me, blocked)
    if res:
        food_move = res.move
    def key(mv):
        return (space_after_move(state, me, mv, blocked, limit=me.length + 20),
                1 if mv == food_move else 0)
    return max(candidates, key=key)


def _apply_guidance(state: State, me: Snake, scores: dict, best: str, blocked: set):
    """Troca a melhor jogada da busca por uma jogada 'guiada' por caminho quando a busca
    não a considera pior que a melhor por mais que a tolerância."""
    best_score = scores[best]
    if abs(best_score) >= WIN_SCORE / 2:       # resultado forçado: não interfere
        return best, "busca"
    tol = GUIDANCE_TOLERANCE

    # Fome: segue o caminho real (BFS) até a comida
    if me.health <= HUNGER_GUIDANCE_HEALTH:
        food = path_to_nearest_food(state, me, blocked)
        if food and food.distance <= me.health and scores.get(food.move, -1e18) > best_score - tol:
            return food.move, "fome: caminho até a comida"

    # Apertado: segue a própria cauda (nunca fica sem saída)
    limit = int(CRAMPED_FACTOR * me.length) + 1
    free = flood_fill_count(state.board, me.head, blocked, limit)
    if free < CRAMPED_FACTOR * me.length:
        tail = path_to_tail(state, me, blocked)
        if tail and scores.get(tail.move, -1e18) > best_score - tol:
            return tail.move, "apertado: seguindo a cauda"

    return best, "busca"


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
    result = search_root(state, deadline, candidates)

    if not result.scores:
        move, reason = _heuristic_pick(state, me, candidates, blocked), "heurística (tempo esgotado)"
    else:
        best = max(result.scores, key=result.scores.get)
        move, reason = _apply_guidance(state, me, result.scores, best, blocked)

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
        "color": "#34095C",
        "head": "fang",
        "tail": "small-rattle",
        "version": "5.0.0",
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
    """Rede de segurança: se o motor falhar por qualquer motivo, ainda joga algo
    válido (dentro do tabuleiro e fora de corpos)."""
    try:
        deltas = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        head = state.you.body[0]
        occupied = {(p.x, p.y) for s in state.board.snakes for p in s.body[:-1]}
        occupied |= {(p.x, p.y) for p in state.you.body[:-1]}
        ok = []
        for name, (dx, dy) in deltas.items():
            x, y = head.x + dx, head.y + dy
            if 0 <= x < state.board.width and 0 <= y < state.board.height and (x, y) not in occupied:
                ok.append(name)
        return random.choice(ok or list(deltas))
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
