
from __future__ import annotations

from collections import deque
import logging
from typing import Iterable

from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DIRECTIONS = ("up", "down", "left", "right")
DELTAS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}




def info() -> dict:
    """GET / — informações visuais e metadados da cobra."""
    return {
        "apiversion": "1",
        "author": "PedroAAmaral",          
        "color": "#8B0000",
        "head": "tiger-king",
        "tail": "hook",
        "version": "2.0.0",
    }


def start(state: GameState) -> None:
    """POST /start — início da partida."""
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)


def end(state: GameState) -> None:
    """POST /end — fim da partida."""
    logger.info("FIM DE JOGO após %d turnos", state.turn)


def get_move(state: GameState) -> MoveResponse:
    """POST /move — decide a próxima jogada.

    A função sempre devolve uma das quatro direções aceitas pela API.
    """

    board_w = state.board.width
    board_h = state.board.height

    my_body = [_pos(p) for p in state.you.body]
    if not my_body:
        # Situação defensiva impossível no jogo normal, mas mantém a API válida.
        return MoveResponse(move="up")

    my_head = my_body[0]
    my_tail = my_body[-1]
    my_length = _snake_length(state.you)
    my_health = _snake_health(state.you)

    foods = {_pos(p) for p in getattr(state.board, "food", [])}
    hazards = {_pos(p) for p in getattr(state.board, "hazards", [])}
    hazard_damage = _hazard_damage(state)

    opponents = [
        snake
        for snake in getattr(state.board, "snakes", [])
        if getattr(snake, "id", None) != getattr(state.you, "id", None)
    ]

    # Direção que a cobra está seguindo no momento. É usada só como desempate
    # / suavização do movimento; nunca supera uma direção perigosa.
    current_direction = _current_direction(my_body)

    # Ocupação rígida para verificar colisão imediata:
    # - próprio corpo, exceto a cauda (que normalmente sai no próximo turno)
    # - corpos adversários, exceto cabeça e cauda, que são dinâmicos
    hard_blocked = set(my_body[:-1])
    opponent_soft_tails: set[tuple[int, int]] = set()
    opponent_heads: list[tuple[int, int]] = []

    for enemy in opponents:
        body = [_pos(p) for p in enemy.body]
        if not body:
            continue

        if len(body) == 1:
            opponent_heads.append(body[0])
        else:
            opponent_heads.append(body[0])
            hard_blocked.update(body[1:-1])
            opponent_soft_tails.add(body[-1])

    # Primeiro passamos por uma camada de segurança. Uma direção que perde a
    # cabeça contra uma cobra maior/igual não entra na lista de candidatas normais.
    candidates: list[dict] = []

    for direction in DIRECTIONS:
        candidate = _next_position(my_head, direction)

        if not _inside(candidate, board_w, board_h):
            continue

        # Recuar sobre o pescoço/próprio corpo continua proibido.
        if candidate in hard_blocked:
            continue

        # Colidir de frente com corpo adversário é morte certa. Cabeça e cauda
        # são tratadas separadamente por serem dinâmicas.
        if candidate in _opponent_hard_body(opponents):
            continue

        # Entrar no corpo da própria cauda é tratado como possível porque ela
        # normalmente será removida no mesmo turno. Isso não vale para um trecho
        # interno do corpo, que já foi bloqueado acima.

        ate_food = candidate in foods
        health_after = _health_after_move(my_health, candidate, ate_food, hazards, hazard_damage)

        # Se a jogada termina o turno sem comida e zera a vida, é uma jogada perdida.
        if health_after <= 0 and not ate_food:
            continue

        # Head-to-head: se uma adversária maior ou igual puder ir exatamente para
        # a casa que estamos escolhendo, nossa jogada é considerada ruim.
        head_threat, attack_value = _head_to_head_values(
            candidate,
            state.you,
            opponents,
            board_w,
            board_h,
            hard_blocked,
            my_body,
        )

        if head_threat >= 100:
            continue

        # Corpo futuro da nossa cobra após esta jogada. Se houver comida, o
        # tamanho aumenta e a cauda antiga permanece.
        future_my_body = _future_body(my_body, candidate, ate_food)
        future_occupied = set(future_my_body)

        # Área que ainda conseguimos alcançar depois de entrar na casa.
        # Para espaço, cabeças/caudas inimigas são consideradas dinâmicas e não
        # viram paredes rígidas.
        future_static_blocked = set(future_occupied)
        for enemy in opponents:
            enemy_body = [_pos(p) for p in enemy.body]
            if len(enemy_body) >= 2:
                future_static_blocked.update(enemy_body[1:-1])

        free_area = _flood_fill(candidate, future_static_blocked, board_w, board_h)
        mobility = _count_moves(candidate, future_static_blocked, board_w, board_h)

        # Olhamos mais um turno à frente. Isso evita escolher uma casa que parece
        # boa agora, mas transforma-se em beco sem saída na jogada seguinte.
        lookahead_area = _lookahead_space(
            future_my_body,
            opponents,
            foods,
            board_w,
            board_h,
        )

        # Território: estima quantas casas ficam mais próximas de nós do que das
        # cabeças adversárias, usando uma busca simultânea em toda a arena.
        territory = _territory_score(
            candidate,
            opponent_heads,
            future_static_blocked,
            board_w,
            board_h,
        )

        # Comida: usa caminho real na grade, não somente distância Manhattan.
        # A prioridade sobe conforme a saúde fica menor.
        food_score, nearest_food_distance = _food_score(
            candidate,
            my_health,
            foods,
            opponents,
            future_static_blocked,
            board_w,
            board_h,
            ate_food,
        )

        # Risco leve de pegar uma cauda adversária que pode desaparecer, mas que
        # também pode permanecer se a cobra comer naquele turno.
        tail_risk = 8 if candidate in opponent_soft_tails else 0

        # Pressão territorial: se a nossa nova cabeça/corpo reduz as saídas de uma
        # cobra menor, isso cria uma oportunidade de encurralamento.
        pressure_score = _attack_pressure_score(
            future_my_body,
            my_length,
            opponents,
            board_w,
            board_h,
            hard_blocked,
        )

        hazard_penalty = _hazard_penalty(
            candidate,
            ate_food,
            my_health,
            hazards,
            hazard_damage,
        )

        # Parede não é somente "não sair do tabuleiro": ficar encostado demais
        # reduz as opções futuras.
        wall_penalty = _wall_penalty(candidate, board_w, board_h)

        # Pequeno incentivo para manter a direção atual e reduzir zigue-zague.
        straight_bonus = 2 if direction == current_direction else 0

        # Quanto mais espaço, melhor. Em tabuleiros pequenos, a área é uma das
        # métricas mais importantes de sobrevivência.
        area_score = (free_area / max(1, board_w * board_h)) * 70

        mobility_score = mobility * 5
        lookahead_score = (lookahead_area / max(1, board_w * board_h)) * 35
        territory_score = territory * 0.12

        # O perigo de cabeça pode ser negativo (risco) ou zero; attack_value é
        # uma recompensa pequena por pressionar uma adversária menor.
        total_score = (
            area_score
            + mobility_score
            + lookahead_score
            + territory_score
            + food_score
            + attack_value
            + pressure_score
            + straight_bonus
            - head_threat
            - tail_risk
            - hazard_penalty
            - wall_penalty
        )

        candidates.append(
            {
                "move": direction,
                "score": total_score,
                "area": free_area,
                "mobility": mobility,
                "territory": territory,
                "food_distance": nearest_food_distance,
                "attack": attack_value,
                "head_risk": head_threat,
                "hazard": hazard_penalty,
                "wall": wall_penalty,
            }
        )

    # Normalmente chegaremos aqui com pelo menos uma boa jogada.
    if candidates:
        # max() é determinístico: em empate mantém a primeira direção de DIRECTIONS.
        chosen_data = max(candidates, key=lambda item: item["score"])
        chosen = chosen_data["move"]

        logger.debug(
            "MOVE %d -> %s | score=%.2f area=%d mobility=%d territory=%d food_dist=%s head_risk=%.2f",
            state.turn,
            chosen,
            chosen_data["score"],
            chosen_data["area"],
            chosen_data["mobility"],
            chosen_data["territory"],
            chosen_data["food_distance"],
            chosen_data["head_risk"],
        )

        return MoveResponse(move=chosen)

    # -----------------------------------------------------------------------
    # Último recurso: estamos encurralados.
    # Escolhemos a direção com melhor pontuação mesmo sabendo que ela é ruim,
    # em vez de sortear cegamente.
    # -----------------------------------------------------------------------
    fallback_data: list[dict] = []

    for direction in DIRECTIONS:
        candidate = _next_position(my_head, direction)
        if not _inside(candidate, board_w, board_h):
            continue

        if candidate in hard_blocked:
            continue

        # Penalização pesada para colisões claras.
        penalty = 0
        if candidate in _opponent_hard_body(opponents):
            penalty += 1000

        head_threat, attack_value = _head_to_head_values(
            candidate,
            state.you,
            opponents,
            board_w,
            board_h,
            hard_blocked,
            my_body,
        )

        area = _flood_fill(candidate, future_block_for_fallback(candidate, my_body, opponents), board_w, board_h)
        fallback_data.append(
            {
                "move": direction,
                "score": area * 10 + attack_value - head_threat - penalty,
            }
        )

    if fallback_data:
        chosen = max(fallback_data, key=lambda item: item["score"])["move"]
    else:
        # Só ocorre em uma posição inválida/inesperada. Mantém a resposta válida.
        chosen = current_direction or "up"

    logger.info("MOVE %d: situação de emergência -> %s", state.turn, chosen)
    return MoveResponse(move=chosen)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def _pos(point) -> tuple[int, int]:
    return int(point.x), int(point.y)


def _snake_length(snake) -> int:
    value = getattr(snake, "length", None)
    if value is not None:
        return int(value)
    return len(getattr(snake, "body", []))


def _snake_health(snake) -> int:
    value = getattr(snake, "health", 100)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 100


def _hazard_damage(state: GameState) -> int:
    """Lê hazardDamagePerTurn quando disponível; em Standard tende a ser irrelevante."""
    try:
        return int(state.game.ruleset.settings.hazardDamagePerTurn)
    except (AttributeError, TypeError, ValueError):
        return 0


def _next_position(position: tuple[int, int], direction: str) -> tuple[int, int]:
    dx, dy = DELTAS[direction]
    return position[0] + dx, position[1] + dy


def _inside(position: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= position[0] < width and 0 <= position[1] < height


def _future_body(
    my_body: list[tuple[int, int]],
    new_head: tuple[int, int],
    ate_food: bool,
) -> list[tuple[int, int]]:
    if ate_food:
        # Crescimento: [novo_head] + corpo atual.
        return [new_head] + list(my_body)
    return [new_head] + list(my_body[:-1])


def _health_after_move(
    health: int,
    destination: tuple[int, int],
    ate_food: bool,
    hazards: set[tuple[int, int]],
    hazard_damage: int,
) -> int:
    if ate_food:
        # Pela ordem das regras, comer repõe a saúde antes da eliminação por 0.
        return 100

    damage = 1
    if destination in hazards:
        damage += hazard_damage

    return health - damage


def _opponent_hard_body(opponents) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()

    for enemy in opponents:
        body = [_pos(p) for p in enemy.body]
        if len(body) <= 2:
            continue
        blocked.update(body[1:-1])

    return blocked


def _current_direction(my_body: list[tuple[int, int]]) -> str | None:
    if len(my_body) < 2:
        return None

    head = my_body[0]
    neck = my_body[1]
    dx = head[0] - neck[0]
    dy = head[1] - neck[1]

    for direction, (ddx, ddy) in DELTAS.items():
        if (dx, dy) == (ddx, ddy):
            return direction

    return None


def _enemy_move_options(
    enemy,
    board_w: int,
    board_h: int,
    occupied_by_others: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Casas que a cabeça adversária pode tentar ocupar no próximo turno."""
    body = [_pos(p) for p in enemy.body]
    if not body:
        return set()

    head = body[0]
    own_body_except_tail = set(body[:-1])

    options = set()
    for dx, dy in DELTAS.values():
        candidate = head[0] + dx, head[1] + dy
        if not _inside(candidate, board_w, board_h):
            continue
        if candidate in own_body_except_tail:
            continue
        if candidate in occupied_by_others:
            continue
        options.add(candidate)

    return options


def _head_to_head_values(
    candidate: tuple[int, int],
    my_snake,
    opponents,
    board_w: int,
    board_h: int,
    my_hard_blocked: set[tuple[int, int]],
    my_body: list[tuple[int, int]],
) -> tuple[float, float]:
    """Retorna (risco, recompensa de ataque) para a casa candidata."""
    my_length = _snake_length(my_snake)
    risk = 0.0
    attack = 0.0

    # O corpo futuro inclui o novo head da nossa cobra, mas para prever se uma
    # adversária pode entrar nessa casa usamos a ocupação atual + nova cabeça.
    # A nossa cabeça atual continua ocupada depois que nos movemos (ela vira
    # parte do corpo), mas a casa candidata é justamente a casa para a qual
    # a adversária poderia mover a cabeça em um head-to-head.
    occupied_for_enemy = set(my_hard_blocked)

    for enemy in opponents:
        enemy_options = _enemy_move_options(
            enemy,
            board_w,
            board_h,
            occupied_for_enemy,
        )

        if candidate not in enemy_options:
            continue

        enemy_length = _snake_length(enemy)

        if my_length <= enemy_length:
            # Empate ou desvantagem: head-to-head pode nos eliminar.
            risk += 160.0
        else:
            # Se somos maiores, há uma oportunidade real de ganhar o head-to-head.
            attack += 18.0

            # Se a cabeça adversária tinha poucas saídas, estamos exercendo
            # pressão adicional.
            if len(enemy_options) == 1:
                attack += 28.0
            elif len(enemy_options) == 2:
                attack += 10.0

    return risk, attack


def _flood_fill(
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
) -> int:
    if not _inside(start, width, height):
        return 0

    # O ponto de partida representa a cabeça depois do movimento, então ele
    # não pode ser considerado uma parede para a própria busca.
    blocked = set(blocked)
    blocked.discard(start)

    visited = {start}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        for dx, dy in DELTAS.values():
            nxt = x + dx, y + dy
            if not _inside(nxt, width, height):
                continue
            if nxt in blocked or nxt in visited:
                continue

            visited.add(nxt)
            queue.append(nxt)

    return len(visited)


def _count_moves(
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
) -> int:
    count = 0
    for dx, dy in DELTAS.values():
        nxt = start[0] + dx, start[1] + dy
        if _inside(nxt, width, height) and nxt not in blocked:
            count += 1
    return count


def _lookahead_space(
    future_my_body: list[tuple[int, int]],
    opponents,
    foods: set[tuple[int, int]],
    width: int,
    height: int,
) -> int:
    """Avalia o maior espaço alcançável após mais uma jogada."""
    if not future_my_body:
        return 0

    head = future_my_body[0]
    self_blocked = set(future_my_body[:-1])
    opponent_blocked = _opponent_hard_body(opponents)
    occupied_base = self_blocked | opponent_blocked

    best_area = 0

    for dx, dy in DELTAS.values():
        nxt = head[0] + dx, head[1] + dy
        if not _inside(nxt, width, height):
            continue
        if nxt in occupied_base:
            continue

        ate_food = nxt in foods
        second_body = _future_body(future_my_body, nxt, ate_food)
        blocked = set(second_body) | opponent_blocked

        area = _flood_fill(nxt, blocked, width, height)
        if area > best_area:
            best_area = area

    return best_area


def _attack_pressure_score(
    future_my_body: list[tuple[int, int]],
    my_length: int,
    opponents,
    width: int,
    height: int,
    current_my_blocked: set[tuple[int, int]],
) -> float:
    """Recompensa a redução das saídas de adversárias menores."""
    if not future_my_body or not opponents:
        return 0.0

    future_occupied = set(future_my_body)
    score = 0.0

    for enemy in opponents:
        enemy_length = _snake_length(enemy)

        # Evitamos incentivar confronto corporal com cobra maior.
        if my_length <= enemy_length:
            continue

        before = _enemy_move_options(
            enemy,
            width,
            height,
            current_my_blocked,
        )
        after = _enemy_move_options(
            enemy,
            width,
            height,
            future_occupied,
        )

        reduction = max(0, len(before) - len(after))
        if reduction == 0:
            continue

        # Tirar duas saídas é uma pressão real; zerar as saídas é ainda melhor.
        score += reduction * 5.0
        if len(after) == 0:
            score += 20.0

    return min(score, 45.0)


def _territory_score(
    my_start: tuple[int, int],
    enemy_heads: Iterable[tuple[int, int]],
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
) -> int:
    """Estima território usando BFS simultâneo (nós x cabeças inimigas)."""
    if not _inside(my_start, width, height):
        return 0

    blocked = set(blocked)
    blocked.discard(my_start)

    # owner: 0 = nós, 1 = adversário
    owner: dict[tuple[int, int], int] = {my_start: 0}
    queue = deque([(my_start, 0)])

    for enemy_head in enemy_heads:
        if not _inside(enemy_head, width, height):
            continue
        if enemy_head in blocked:
            continue
        if enemy_head in owner:
            # Se começarmos exatamente na mesma casa, é um caso de disputa de
            # território; a célula não vale como território exclusivo nosso.
            continue
        owner[enemy_head] = 1
        queue.append((enemy_head, 1))

    ours = 0
    enemies = 0

    while queue:
        position, who = queue.popleft()

        if who == 0:
            ours += 1
        else:
            enemies += 1

        x, y = position
        for dx, dy in DELTAS.values():
            nxt = x + dx, y + dy
            if not _inside(nxt, width, height):
                continue
            if nxt in blocked or nxt in owner:
                continue

            owner[nxt] = who
            queue.append((nxt, who))

    return ours - enemies // 2


def _nearest_food_distance(
    start: tuple[int, int],
    foods: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
) -> int | None:
    if not foods:
        return None
    if start in foods:
        return 0

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        position, distance = queue.popleft()
        x, y = position

        for dx, dy in DELTAS.values():
            nxt = x + dx, y + dy
            if not _inside(nxt, width, height):
                continue
            if nxt in visited or nxt in blocked:
                continue
            if nxt in foods:
                return distance + 1

            visited.add(nxt)
            queue.append((nxt, distance + 1))

    return None


def _food_score(
    candidate: tuple[int, int],
    health: int,
    foods: set[tuple[int, int]],
    opponents,
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
    ate_food: bool,
) -> tuple[float, int | None]:
    if ate_food:
        # Comer agora vale bastante, especialmente quando a saúde está baixa.
        immediate = 55.0 if health <= 30 else 35.0 if health <= 60 else 14.0
        return immediate, 0

    distance = _nearest_food_distance(
        candidate,
        foods,
        blocked,
        width,
        height,
    )

    if distance is None:
        return 0.0, None

    if health <= 25:
        weight = 82.0
    elif health <= 45:
        weight = 58.0
    elif health <= 70:
        weight = 34.0
    else:
        weight = 14.0

    # A recompensa cai rapidamente com a distância. Assim, uma comida que está
    # a um passo não é confundida com uma comida distante que só parece próxima
    # pela área disponível do tabuleiro.
    score = weight / (distance + 1)

    # Se a comida está muito longe para a saúde atual, não queremos persegui-la
    # cegamente. Ainda existe algum valor para ela, mas reduzimos bastante.
    if distance >= health:
        score *= 0.20
    elif distance >= health - 2:
        score *= 0.55

    # Competição pela comida: se um inimigo está muito perto do mesmo alimento,
    # a recompensa diminui porque a corrida pode gerar uma situação perigosa.
    if foods:
        contest_penalty = 0.0
        for food in foods:
            food_distance = _manhattan(candidate, food)
            if food_distance != distance:
                continue

            enemy_closest = None
            for enemy in opponents:
                enemy_body = [_pos(p) for p in enemy.body]
                if not enemy_body:
                    continue
                d = _manhattan(enemy_body[0], food)
                if enemy_closest is None or d < enemy_closest:
                    enemy_closest = d

            if enemy_closest is not None and enemy_closest <= max(1, food_distance + 1):
                contest_penalty = max(contest_penalty, 0.45)

        score *= (1.0 - contest_penalty)

    return score, distance


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _hazard_penalty(
    candidate: tuple[int, int],
    ate_food: bool,
    health: int,
    hazards: set[tuple[int, int]],
    hazard_damage: int,
) -> float:
    if candidate not in hazards or ate_food:
        return 0.0

    damage = 1 + hazard_damage
    remaining = health - damage

    if remaining <= 0:
        return 250.0

    # Mesmo sobrevivendo, hazards consomem saúde e merecem penalidade.
    return 12.0 + hazard_damage * 0.8


def _wall_penalty(
    position: tuple[int, int],
    width: int,
    height: int,
) -> float:
    x, y = position
    nearest = min(x, y, width - 1 - x, height - 1 - y)

    if nearest == 0:
        return 14.0
    if nearest == 1:
        return 6.0
    return 0.0


def future_block_for_fallback(
    candidate: tuple[int, int],
    my_body: list[tuple[int, int]],
    opponents,
) -> set[tuple[int, int]]:
    blocked = set(_future_body(my_body, candidate, False))
    for enemy in opponents:
        body = [_pos(p) for p in enemy.body]
        if len(body) >= 2:
            blocked.update(body[1:-1])
    return blocked
