"""Nível de perigo a partir da distância, da posição e da classe."""

from __future__ import annotations

from obstaclesense.classes import VEHICLE_LABELS

LEVEL_RANK = {"high": 0, "medium": 1, "low": 2}


def assess_danger(
    distance_m: float,
    direction: str,
    label: str,
    approaching: bool = False,
    proximity_m: float = 2.0,
    collision_threshold: float = 0.7,
) -> tuple[str, float]:
    """Devolve (low|medium|high, pontuação de 0 a 1).

    `collision_warning_threshold` é o corte da pontuação para o nível alto.
    Veículos e objetos no centro do caminho pesam mais. Aproximação soma
    pontos para o rastreador poder elevar o alerta.
    """
    proximity = max(0.2, float(proximity_m))
    vehicle = label in VEHICLE_LABELS
    in_path = direction == "center"
    very_close = distance_m <= proximity * 0.5
    # A folga evita que 2,0 m e 2,1 m caiam em níveis opostos.
    close = distance_m <= proximity * 1.2
    near = distance_m <= proximity * 3.0

    if very_close and in_path:
        score = 0.92
    elif very_close:
        score = 0.78
    elif close and in_path and vehicle:
        score = 0.88
    elif close and in_path:
        score = 0.74
    elif close and vehicle:
        score = 0.76
    elif close:
        score = 0.52
    elif near and in_path and vehicle:
        score = 0.62
    elif near and in_path:
        score = 0.48
    elif near and vehicle:
        score = 0.42
    else:
        score = 0.22

    if approaching:
        score = min(1.0, score + 0.12)

    threshold = max(0.05, min(1.0, float(collision_threshold)))
    if score >= threshold:
        level = "high"
    elif score >= threshold * 0.6:
        level = "medium"
    else:
        level = "low"
    return level, round(score, 4)


def sort_obstacles(obstacles: list) -> list:
    """Mais perigoso primeiro; empate vai para o mais próximo e para veículos."""

    def key(item):
        return (
            LEVEL_RANK.get(item.danger_level, 3),
            -float(getattr(item, "danger_score", 0.0)),
            item.distance_meters,
        )

    return sorted(obstacles, key=key)
