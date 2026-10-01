
from __future__ import annotations

from collections import deque
import logging
import random
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


# ---------------------------------------------------------------------------
# API obrigatória do template
# ---------------------------------------------------------------------------


def info() -> dict:
    """GET / — informações visuais e metadados da cobra."""
    return {
        "apiversion": "1",
        "author": "PedroAAmaral",  # seu usuário do Battlesnake
        "color": "#8B0000",
        "head": "tiger-king",
        "tail": "hook",
        "version": "3.1.0",
    }


def start(state: GameState) -> None:
    """POST /start — início da partida."""
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)


def end(state: GameState) -> None:
    """POST /end — fim da partida."""
    logger.info("FIM DE JOGO após %d turnos", state.turn)


def get_move(state: GameState) -> MoveResponse:
    """POST /move — devolve a jogada, com rede de segurança.

    Se qualquer parte da estratégia lançar uma exceção, o jogo não fica sem
    resposta: caímos numa jogada simples (dentro do tabuleiro e fora de corpos).
    """
    try:
        return _choose_move(state)
    except Exception:
        logger.exception("MOVE %s: erro na estratégia; usando fallback simples",
                         getattr(state, "turn", "?"))
        return MoveResponse(move=_simple_fallback(state))


def _simple_fallback(state: GameState) -> str:
    """Jogada de último recurso, sem depender do resto da estratégia."""
    try:
        head = _pos(state.you.body[0])
        occupied = {_pos(p) for s in state.board.snakes for p in s.body[:-1]}
        occupied |= {_pos(p) for p in state.you.body[:-1]}
        ok = [
            d for d in DIRECTIONS
            if _inside(_next_position(head, d), state.board.width, state.board.height)
            and _next_position(head, d) not in occupied
        ]
        return random.choice(ok or list(DIRECTIONS))
    except Exception:
        return "up"


def _choose_move(state: GameState) -> MoveResponse:
    """Escolhe a jogada usando heurística + lookahead adversarial.

    A ideia é evoluir de:
        "qual jogada parece melhor agora?"
    para:
        "qual jogada continua boa quando um adversário responde da forma mais
        incômoda possível?"

    O lookahead é propositalmente curto para respeitar a janela de resposta do
    campeonato. A camada tática continua barata porque simula apenas respostas
    de um adversário por vez e usa BFS em um tabuleiro pequeno.
    """

    board_w = state.board.width
    board_h = state.board.height

    my_body = [_pos(p) for p in state.you.body]
    if not my_body:
        return MoveResponse(move="up")

    my_head = my_body[0]
    my_health = _snake_health(state.you)
    my_length = len(my_body)

    foods = {_pos(p) for p in getattr(state.board, "food", [])}
    hazards = {_pos(p) for p in getattr(state.board, "hazards", [])}
    hazard_damage = _hazard_damage(state)

    opponents = [
        snake
        for snake in getattr(state.board, "snakes", [])
        if getattr(snake, "id", None) != getattr(state.you, "id", None)
    ]

    current_direction = _current_direction(my_body)

    # Corpo atual dos adversários. Mantemos cabeça e cauda fora do conjunto
    # rígido porque são partes dinâmicas na resolução do turno.
    opponent_soft_tails: set[tuple[int, int]] = set()
    opponent_heads: list[tuple[int, int]] = []
    opponent_hard_body: set[tuple[int, int]] = set()
    opponent_head_cells: set[tuple[int, int]] = set()

    for enemy in opponents:
        body = [_pos(p) for p in enemy.body]
        if not body:
            continue
        opponent_heads.append(body[0])
        if len(body) >= 2:
            opponent_soft_tails.add(body[-1])
            opponent_hard_body.update(body[1:-1])
            # A cabeça atual vira o pescoço no próximo turno.
            opponent_head_cells.add(body[0])

    # -----------------------------------------------------------------------
    # 1) Geração de jogadas legalmente possíveis.
    # -----------------------------------------------------------------------
    candidates: list[dict] = []

    for direction in DIRECTIONS:
        candidate = _next_position(my_head, direction)

        if not _inside(candidate, board_w, board_h):
            continue

        # Não voltar sobre o pescoço nem entrar em trecho interno do próprio corpo.
        if candidate in set(my_body[:-1]):
            continue

        # Entrar na própria cauda só é seguro quando ela realmente vai sair. Se
        # houver comida na cauda, a cobra cresce e a cauda permanece.
        if len(my_body) >= 2 and candidate == my_body[-1] and candidate in foods:
            continue

        # Corpo interno adversário é colisão direta. Cabeça e cauda passam pela
        # camada tática porque podem gerar head-to-head / tail movement.
        if candidate in opponent_hard_body:
            continue

        # Entrar na casa onde a cabeça de um rival está agora é colisão CERTA:
        # ela vira o pescoço dele depois do movimento (nunca é um head-to-head).
        if candidate in opponent_head_cells:
            continue

        ate_food = candidate in foods
        health_after = _health_after_move(
            my_health, candidate, ate_food, hazards, hazard_damage
        )

        if health_after <= 0 and not ate_food:
            continue

        # Head-to-head imediato. Se uma cobra pelo menos tão grande pode entrar
        # na mesma casa, a jogada deixa de ser candidata normal.
        immediate_head_risk, immediate_attack = _head_to_head_values(
            candidate,
            state.you,
            opponents,
            board_w,
            board_h,
            set(my_body[:-1]),
            my_body,
        )

        # Jogada com risco de head-to-head perdido NÃO é descartada: vira "plano B".
        # Uma jogada arriscada (talvez morra) é melhor que uma sem saída (morre).
        risky_head = immediate_head_risk >= 100

        future_my_body = _future_body(my_body, candidate, ate_food)
        future_blocked = set(future_my_body[:-1]) | opponent_hard_body

        free_area = _flood_fill(
            candidate, future_blocked, board_w, board_h
        )
        mobility = _count_moves(
            candidate, future_blocked, board_w, board_h
        )
        tail_access = _tail_access_score(
            future_my_body,
            future_blocked,
            board_w,
            board_h,
        )
        bottleneck_penalty = _bottleneck_penalty(
            free_area,
            mobility,
            len(future_my_body),
            board_w,
            board_h,
        )
        lookahead_area = _lookahead_space(
            future_my_body,
            opponents,
            foods - ({candidate} if ate_food else set()),
            board_w,
            board_h,
        )
        territory = _territory_score(
            candidate,
            opponent_heads,
            future_blocked,
            board_w,
            board_h,
        )

        food_score, nearest_food_distance = _food_score(
            candidate,
            my_health,
            foods,
            opponents,
            future_blocked,
            board_w,
            board_h,
            ate_food,
        )

        tail_risk = 8.0 if candidate in opponent_soft_tails else 0.0
        pressure_score = _attack_pressure_score(
            future_my_body,
            my_length,
            opponents,
            board_w,
            board_h,
            set(my_body[:-1]),
        )
        hazard_penalty = _hazard_penalty(
            candidate,
            ate_food,
            my_health,
            hazards,
            hazard_damage,
        )
        wall_penalty = _wall_penalty(candidate, board_w, board_h)
        straight_bonus = 2.0 if direction == current_direction else 0.0

        # Corrida pela comida: uma comida muito disputada vale menos que parece.
        food_race_penalty = _food_race_penalty(
            candidate,
            foods - ({candidate} if ate_food else set()),
            opponents,
            future_blocked,
            board_w,
            board_h,
            my_health,
        )

        area_score = (free_area / max(1, board_w * board_h)) * 70.0
        mobility_score = mobility * 5.0
        lookahead_score = (lookahead_area / max(1, board_w * board_h)) * 35.0
        territory_score = territory * 0.12

        base_score = (
            area_score
            + mobility_score
            + lookahead_score
            + territory_score
            + food_score
            + immediate_attack
            + pressure_score
            + tail_access
            + straight_bonus
            - immediate_head_risk
            - tail_risk
            - hazard_penalty
            - wall_penalty
            - food_race_penalty
            - bottleneck_penalty
        )

        # -------------------------------------------------------------------
        # 2) Lookahead adversarial.
        # -------------------------------------------------------------------
        tactical_score = _adversarial_lookahead(
            my_body=my_body,
            my_health=my_health,
            my_length=my_length,
            direction=direction,
            opponents=opponents,
            foods=foods,
            hazards=hazards,
            hazard_damage=hazard_damage,
            width=board_w,
            height=board_h,
        )

        # A heurística atual continua tendo peso maior: o lookahead é uma camada
        # de correção tática, e não uma busca profunda que possa estourar o timeout.
        total_score = base_score + tactical_score

        candidates.append(
            {
                "move": direction,
                "risky": risky_head,
                "score": total_score,
                "base_score": base_score,
                "tactical_score": tactical_score,
                "area": free_area,
                "mobility": mobility,
                "territory": territory,
                "food_distance": nearest_food_distance,
                "attack": immediate_attack,
                "head_risk": immediate_head_risk,
                "hazard": hazard_penalty,
                "wall": wall_penalty,
            }
        )

    if candidates:
        # DIRECTIONS é uma tupla fixa, então o empate permanece determinístico.
        # Jogadas sem risco de head-to-head têm prioridade; as arriscadas só
        # entram na disputa quando não existe nenhuma segura.
        safe_candidates = [c for c in candidates if not c["risky"]]
        pool = safe_candidates or candidates
        chosen_data = max(pool, key=lambda item: item["score"])
        chosen = chosen_data["move"]

        logger.info(
            "MOVE %d -> %s | total=%.2f base=%.2f tactical=%.2f area=%d mob=%d territory=%d food=%s head_risk=%.2f",
            state.turn,
            chosen,
            chosen_data["score"],
            chosen_data["base_score"],
            chosen_data["tactical_score"],
            chosen_data["area"],
            chosen_data["mobility"],
            chosen_data["territory"],
            chosen_data["food_distance"],
            chosen_data["head_risk"],
        )
        return MoveResponse(move=chosen)

    # -----------------------------------------------------------------------
    # Último recurso. O cenário normal nunca chega aqui, mas, se todas as
    # direções seguras desaparecerem, escolhemos deterministicamente a menos
    # ruim em vez de usar aleatoriedade.
    # -----------------------------------------------------------------------
    fallback: list[tuple[float, str]] = []

    for direction in DIRECTIONS:
        candidate = _next_position(my_head, direction)
        if not _inside(candidate, board_w, board_h):
            continue

        score = 0.0
        if candidate in set(my_body[:-1]):
            score -= 1000.0
        if candidate in opponent_hard_body:
            score -= 1200.0
        if candidate in hazards:
            score -= 50.0

        future_body = _future_body(my_body, candidate, False)
        area = _flood_fill(
            candidate,
            set(future_body[:-1]) | opponent_hard_body,
            board_w,
            board_h,
        )
        score += area * 10.0
        fallback.append((score, direction))

    chosen = max(fallback)[1] if fallback else (current_direction or "up")
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
        # Se há hazards mas a regra não veio no estado, usa o padrão do Royale (14);
        # subestimar o dano faz a cobra entrar em zonas que a matam de fome.
        return 14 if getattr(state.board, "hazards", None) else 0


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




def _tail_access_score(
    body: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
) -> float:
    """Premia posições que mantêm um caminho de fuga até a própria cauda."""
    if len(body) < 2:
        return 0.0

    head = body[0]
    tail = body[-1]
    if head == tail:
        return 0.0

    queue = deque([(head, 0)])
    visited = {head}
    blocked_local = set(blocked)
    blocked_local.discard(tail)
    blocked_local.discard(head)

    while queue:
        position, distance = queue.popleft()
        if position == tail:
            # Quanto menor a distância até a cauda, mais opções temos para
            # prolongar a partida em situações de aperto.
            return max(3.0, 18.0 - distance * 1.5)

        x, y = position
        for dx, dy in DELTAS.values():
            nxt = x + dx, y + dy
            if not _inside(nxt, width, height):
                continue
            if nxt in visited or nxt in blocked_local:
                continue
            visited.add(nxt)
            queue.append((nxt, distance + 1))

    return -15.0


def _bottleneck_penalty(
    area: int,
    mobility: int,
    length: int,
    width: int,
    height: int,
) -> float:
    """Penaliza entrar em regiões pequenas com poucas saídas.

    O critério principal é o espaço comparado ao TAMANHO da cobra: se não cabe o
    próprio corpo, a cobra está presa, seja ela curta ou longa.
    """
    if mobility == 0:
        return 80.0

    penalty = 0.0

    if mobility == 1:
        penalty += 24.0
    elif mobility == 2 and area <= max(5, length * 2):
        penalty += 15.0

    if area < length:
        penalty += 40.0          # não cabe o corpo: praticamente um beco fatal
    elif area < length * 1.5:
        penalty += 15.0          # cabe, mas sobra pouco

    return min(70.0, penalty)


def _food_race_penalty(
    candidate: tuple[int, int],
    foods: set[tuple[int, int]],
    opponents,
    blocked: set[tuple[int, int]],
    width: int,
    height: int,
    my_health: int,
) -> float:
    """Reduz a atração de comida que um adversário deve alcançar primeiro.

    A distância usada é BFS, então paredes e corpos entram na conta. Isso evita
    tratar uma comida que está "perto em linha reta", mas atrás de um corredor,
    como uma corrida simples.
    """
    if not foods or not opponents:
        return 0.0

    my_distance = _nearest_food_distance(
        candidate, foods, blocked, width, height
    )
    if my_distance is None:
        return 0.0

    penalty = 0.0

    for food in foods:
        # Só analisamos alimentos que podem corresponder ao caminho mais curto
        # encontrado a partir da nossa candidata.
        food_distance = _nearest_food_distance(
            candidate, {food}, blocked, width, height
        )
        if food_distance is None or food_distance != my_distance:
            continue

        for enemy in opponents:
            body = [_pos(p) for p in getattr(enemy, "body", [])]
            if not body:
                continue

            enemy_head = body[0]
            enemy_blocked = set()
            if len(body) >= 2:
                enemy_blocked.update(body[:-1])

            # Não usamos a nossa futura cabeça como bloqueio para o cálculo da
            # corrida; a competição pela casa é tratada como risco tático.
            enemy_distance = _nearest_food_distance(
                enemy_head,
                {food},
                enemy_blocked,
                width,
                height,
            )

            if enemy_distance is None:
                continue

            if enemy_distance < food_distance:
                penalty += 11.0
            elif enemy_distance == food_distance:
                penalty += 6.0
            elif enemy_distance == food_distance + 1:
                penalty += 2.5

    # A comida é mais importante conforme a saúde cai, então reduzimos a
    # penalização quando estamos perto de morrer de fome.
    if my_health <= 25:
        penalty *= 0.45
    elif my_health <= 45:
        penalty *= 0.70

    return min(30.0, penalty)


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


# ---------------------------------------------------------------------------
# Lookahead adversarial / simulação tática
# ---------------------------------------------------------------------------


def _adversarial_lookahead(
    *,
    my_body: list[tuple[int, int]],
    my_health: int,
    my_length: int,
    direction: str,
    opponents,
    foods: set[tuple[int, int]],
    hazards: set[tuple[int, int]],
    hazard_damage: int,
    width: int,
    height: int,
) -> float:
    """Faz um minimax tático raso: nós escolhemos a jogada, um inimigo responde.

    Não tenta enumerar todas as combinações simultâneas de todas as cobras,
    porque isso explode rapidamente. Em vez disso, para cada inimigo avaliamos
    suas respostas e consideramos a pior resposta individual. Isso já captura
    boa parte dos problemas que uma avaliação puramente local não percebe:
    corredor fechado, fuga cortada, pressão de cabeça e perda do acesso à comida.
    """
    destination = _next_position(my_body[0], direction)
    ate_food = destination in foods

    after_me = _future_body(my_body, destination, ate_food)
    after_me_health = _health_after_move(
        my_health, destination, ate_food, hazards, hazard_damage
    )

    if after_me_health <= 0 and not ate_food:
        return -900.0

    remaining_food = foods - ({destination} if ate_food else set())
    if not opponents:
        # Mesmo sem inimigos, valorizamos a qualidade da próxima posição.
        return 0.35 * _next_state_value(
            after_me,
            after_me_health,
            opponents,
            remaining_food,
            hazards,
            hazard_damage,
            width,
            height,
        )

    worst_response = float("inf")
    saw_response = False

    # Os adversários são avaliados separadamente. A menor nota é a ameaça que
    # conseguimos identificar com a nossa visão rasa.
    for enemy_index, enemy in enumerate(opponents):
        enemy_body = [_pos(p) for p in getattr(enemy, "body", [])]
        if not enemy_body:
            continue

        enemy_health = _snake_health(enemy)
        enemy_length = len(enemy_body)
        enemy_head = enemy_body[0]
        enemy_reverse = _current_direction(enemy_body)

        for enemy_direction in DIRECTIONS:
            if enemy_reverse is not None and enemy_direction == _opposite(enemy_reverse):
                continue

            enemy_destination = _next_position(enemy_head, enemy_direction)
            if not _inside(enemy_destination, width, height):
                continue

            # Movimento adversário sem a nossa resposta. Colisões são resolvidas
            # depois; isso permite considerar head-to-head e entradas em corpo.
            enemy_ate = enemy_destination in remaining_food
            after_enemy = _future_body(enemy_body, enemy_destination, enemy_ate)
            after_enemy_health = _health_after_move(
                enemy_health,
                enemy_destination,
                enemy_ate,
                hazards,
                hazard_damage,
            )

            # As outras cobras continuam estáticas nesta aproximação. Para a
            # nossa sobrevivência, seus corpos internos são obstáculos.
            other_bodies = set()
            for j, other in enumerate(opponents):
                if j == enemy_index:
                    continue
                body = [_pos(p) for p in getattr(other, "body", [])]
                if len(body) >= 2:
                    other_bodies.update(body[1:-1])

            ours_alive, enemy_alive = _resolve_tactical_collision(
                our_body=after_me,
                our_health=after_me_health,
                enemy_body=after_enemy,
                enemy_health=after_enemy_health,
                enemy_length=len(after_enemy),
                static_bodies=other_bodies,
                width=width,
                height=height,
            )

            saw_response = True

            if not ours_alive:
                response_value = -1000.0
            else:
                response_value = _next_state_value(
                    after_me,
                    after_me_health,
                    [
                        _EnemySnapshot(
                            body=after_enemy,
                            health=after_enemy_health,
                            length=len(after_enemy),
                            alive=enemy_alive,
                        )
                    ],
                    remaining_food - ({enemy_destination} if enemy_ate else set()),
                    hazards,
                    hazard_damage,
                    width,
                    height,
                )

                # Se nós comemos e ficamos maiores, uma eliminação imediata do
                # adversário é boa. Se o adversário sobreviveu e está adjacente,
                # o próximo turno ainda pode representar uma disputa de cabeça.
                if not enemy_alive:
                    response_value += 75.0
                else:
                    response_value -= _future_head_threat(
                        after_me,
                        enemy_body=[after_enemy[0]],   # só a CABEÇA, não o corpo todo
                        our_length=len(after_me),
                        enemy_length=len(after_enemy),
                        width=width,
                        height=height,
                    )

            if response_value < worst_response:
                worst_response = response_value

    if not saw_response:
        return 0.35 * _next_state_value(
            after_me,
            after_me_health,
            opponents,
            remaining_food,
            hazards,
            hazard_damage,
            width,
            height,
        )

    # Peso moderado para não destruir a heurística principal em partidas com
    # muitas cobras. Ainda assim, uma linha que perde em um cenário de resposta
    # óbvia recebe forte penalização.
    return max(-260.0, min(95.0, worst_response * 0.70))


class _EnemySnapshot:
    """Pequeno contêiner interno usado só na simulação."""

    __slots__ = ("body", "health", "length", "alive")

    def __init__(self, body, health, length, alive=True):
        self.body = body
        self.health = health
        self.length = length
        self.alive = alive


def _opposite(direction: str) -> str:
    return {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }[direction]


def _resolve_tactical_collision(
    *,
    our_body: list[tuple[int, int]],
    our_health: int,
    enemy_body: list[tuple[int, int]],
    enemy_health: int,
    enemy_length: int,
    static_bodies: set[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[bool, bool]:
    """Resolve só o que interessa para o lookahead: quem sobrevive ao turno."""
    if not our_body or not enemy_body:
        return bool(our_body), bool(enemy_body)

    our_head = our_body[0]
    enemy_head = enemy_body[0]
    our_length = len(our_body)

    our_alive = True
    enemy_alive = True

    if our_health <= 0:
        our_alive = False
    if enemy_health <= 0:
        enemy_alive = False

    # Colisão cabeça-a-cabeça: maior sobrevive; empate elimina as duas.
    if our_head == enemy_head:
        if our_length > enemy_length:
            enemy_alive = False
        elif our_length < enemy_length:
            our_alive = False
        else:
            our_alive = False
            enemy_alive = False

    # Cabeça em corpo.
    if our_alive and our_head in set(enemy_body[1:]):
        our_alive = False
    if enemy_alive and enemy_head in set(our_body[1:]):
        enemy_alive = False

    # Colisão com o PRÓPRIO corpo (um rival não joga para dentro de si mesmo).
    if our_alive and our_head in set(our_body[1:]):
        our_alive = False
    if enemy_alive and enemy_head in set(enemy_body[1:]):
        enemy_alive = False

    # Colisões com obstáculos estáticos / corpos de outras cobras.
    static = set(static_bodies)
    if our_alive and our_head in static:
        our_alive = False
    if enemy_alive and enemy_head in static:
        enemy_alive = False

    # Limites. Já deveriam ter sido filtrados, mas a checagem torna a função
    # segura para testes unitários e futuras mudanças.
    if not _inside(our_head, width, height):
        our_alive = False
    if not _inside(enemy_head, width, height):
        enemy_alive = False

    return our_alive, enemy_alive


def _next_state_value(
    my_body: list[tuple[int, int]],
    health: int,
    opponents,
    foods: set[tuple[int, int]],
    hazards: set[tuple[int, int]],
    hazard_damage: int,
    width: int,
    height: int,
) -> float:
    """Valoriza a posição um turno depois da resposta adversária."""
    if not my_body:
        return -1000.0

    my_head = my_body[0]
    occupied = set(my_body[:-1])

    enemy_bodies = []
    enemy_heads = []
    enemy_hard = set()

    for enemy in opponents:
        if not getattr(enemy, "alive", True):
            continue
        body = list(getattr(enemy, "body", []))
        if not body:
            continue
        enemy_bodies.append(body)
        enemy_heads.append(body[0])
        if len(body) >= 2:
            enemy_hard.update(body[1:-1])

    static_blocked = occupied | enemy_hard

    legal_next: list[tuple[str, tuple[int, int], int, int]] = []
    for direction in DIRECTIONS:
        nxt = _next_position(my_head, direction)
        if not _inside(nxt, width, height):
            continue
        if nxt in occupied:
            continue
        if nxt == my_body[-1] and nxt in foods and len(my_body) >= 2:
            continue
        if nxt in enemy_hard:
            continue

        ate = nxt in foods
        next_body = _future_body(my_body, nxt, ate)
        next_health = _health_after_move(
            health,
            nxt,
            ate,
            hazards,
            hazard_damage,
        )
        if next_health <= 0 and not ate:
            continue

        area = _flood_fill(
            nxt,
            set(next_body[:-1]) | enemy_hard,
            width,
            height,
        )
        mobility = _count_moves(
            nxt,
            set(next_body[:-1]) | enemy_hard,
            width,
            height,
        )
        legal_next.append((direction, nxt, area, mobility))

    if not legal_next:
        return -700.0

    best = -float("inf")
    for direction, nxt, area, mobility in legal_next:
        food_value = 0.0
        nearest = _nearest_food_distance(
            nxt,
            foods,
            static_blocked,
            width,
            height,
        )
        if nearest is not None:
            if health <= 25:
                food_value = 55.0 / (nearest + 1)
            elif health <= 50:
                food_value = 30.0 / (nearest + 1)
            else:
                food_value = 12.0 / (nearest + 1)

        territory = _territory_score(
            nxt,
            enemy_heads,
            set(_future_body(my_body, nxt, nxt in foods)[:-1]) | enemy_hard,
            width,
            height,
        )

        head_risk = _future_head_threat(
            _future_body(my_body, nxt, nxt in foods),
            enemy_body=enemy_heads,
            our_length=len(_future_body(my_body, nxt, nxt in foods)),
            enemy_lengths=[len(body) for body in enemy_bodies],
            width=width,
            height=height,
        ) if enemy_bodies else 0.0

        value = (
            area * 1.65
            + mobility * 9.0
            + territory * 0.30
            + food_value
            - head_risk
        )

        if nxt in hazards and nxt not in foods:
            value -= 12.0 + hazard_damage * 0.8

        best = max(best, value)

    return best


def _future_head_threat(
    my_body: list[tuple[int, int]],
    enemy_body,
    our_length: int,
    enemy_length=None,
    enemy_lengths=None,
    width: int = 0,
    height: int = 0,
) -> float:
    """Penaliza vizinhança imediata de cabeças que podem ganhar o próximo confronto."""
    if not my_body:
        return 200.0

    my_head = my_body[0]

    if enemy_body is None:
        return 0.0

    if enemy_body and isinstance(enemy_body[0], tuple):
        enemy_heads = list(enemy_body)
    else:
        enemy_heads = []
        for item in enemy_body:
            body = getattr(item, "body", None)
            if body:
                enemy_heads.append(body[0])
            elif isinstance(item, tuple):
                enemy_heads.append(item)

    lengths: list[int]
    if enemy_lengths is not None:
        lengths = list(enemy_lengths)
    elif enemy_length is not None:
        lengths = [int(enemy_length)] * len(enemy_heads)
    else:
        lengths = [1] * len(enemy_heads)

    threat = 0.0
    for index, head in enumerate(enemy_heads):
        distance = abs(my_head[0] - head[0]) + abs(my_head[1] - head[1])
        length = lengths[min(index, len(lengths) - 1)] if lengths else 1

        if distance == 1:
            if length >= our_length:
                threat += 55.0
            else:
                threat -= 8.0
        elif distance == 2 and length >= our_length:
            # Menor que o risco adjacente, mas ainda é uma disputa que pode
            # aparecer no próximo turno dependendo da geometria.
            threat += 12.0

    return min(90.0, max(-20.0, threat))


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
