"""Frases de orientação faladas somente em português do Brasil."""

from __future__ import annotations

from dataclasses import dataclass

from obstaclesense.classes import spoken_name

_UNITS = [
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
    "onze",
    "doze",
    "treze",
    "quatorze",
    "quinze",
    "dezesseis",
    "dezessete",
    "dezoito",
    "dezenove",
]

_PLACE = {
    "left": "à sua esquerda",
    "right": "à sua direita",
    "center": "à sua frente",
}

CLEAR_PATH = "Nenhum obstáculo à frente."


@dataclass
class PhraseObstacle:
    label: str
    distance_m: float
    direction: str
    danger_level: str
    approaching: bool = False


def number_words(value: int) -> str:
    if value < 0:
        return str(value)
    if value < 20:
        return _UNITS[value]
    if value < 30:
        if value == 20:
            return "vinte"
        return f"vinte e {_UNITS[value - 20]}"
    if value == 30:
        return "trinta"
    return str(value)


def distance_words(meters: float) -> str:
    """Arredonda para a fala: 'dois metros', 'um metro e meio'."""
    if meters < 0.75:
        return "menos de um metro"
    nearest_half = round(meters * 2.0) / 2.0
    if abs(nearest_half - round(nearest_half)) == 0.5 and abs(meters - nearest_half) <= 0.2:
        whole = int(nearest_half)
        if whole <= 0:
            return "meio metro"
        unit = "metro" if whole == 1 else "metros"
        return f"{number_words(whole)} {unit} e meio"
    whole = max(1, int(round(meters)))
    if whole == 1:
        return "um metro"
    return f"{number_words(whole)} metros"


def choose_action(primary: PhraseObstacle, others: list[PhraseObstacle], proximity_m: float) -> str:
    """Desvio para o lado livre. Os dois lados ocupados pedem parada."""
    if primary.danger_level == "low":
        return ""

    limit = proximity_m * 1.5

    def busy(side: str) -> bool:
        if primary.direction == side and primary.danger_level in {"medium", "high"}:
            return True
        # Qualquer obstáculo perto ocupa o lado, mesmo com perigo baixo.
        return any(item.direction == side and item.distance_m <= limit for item in others)

    left_busy = busy("left")
    right_busy = busy("right")
    if primary.direction == "left":
        return "Pare." if right_busy else "Desvie para a direita."
    if primary.direction == "right":
        return "Pare." if left_busy else "Desvie para a esquerda."
    if left_busy and right_busy:
        return "Pare."
    if right_busy and not left_busy:
        return "Desvie para a esquerda."
    if left_busy and not right_busy:
        return "Desvie para a direita."
    return "Desvie para a direita."


def build_guidance(obstacles: list[PhraseObstacle], proximity_m: float = 2.0) -> str:
    """Uma frase curta sobre o obstáculo mais perigoso, já ordenado."""
    if not obstacles:
        return CLEAR_PATH
    primary = obstacles[0]
    article, name = spoken_name(primary.label)
    place = _PLACE.get(primary.direction, "à sua frente")
    action = choose_action(primary, obstacles[1:], proximity_m)

    if primary.distance_m < 0.75:
        sentence = f"Cuidado! Existe {article} {name} muito perto {place}."
    elif primary.approaching:
        dist = distance_words(primary.distance_m)
        sentence = (
            f"Atenção! Existe {article} {name} se aproximando, "
            f"aproximadamente {dist} {place}."
        )
    else:
        dist = distance_words(primary.distance_m)
        sentence = f"Existe {article} {name} aproximadamente {dist} {place}."

    if action:
        return f"{sentence} {action}"
    return sentence
