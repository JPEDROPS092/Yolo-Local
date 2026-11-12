"""Distância em metros: tamanho aparente + plano do chão, com fusão de profundidade."""

from __future__ import annotations

import math

from obstaclesense.classes import REAL_SIZE_M

MIN_DISTANCE_M = 0.3
MAX_DISTANCE_M = 30.0


def focal_length_px(image_size_px: float, fov_degrees: float) -> float:
    """Distância focal em pixels a partir do campo de visão."""
    fov = max(1.0, min(179.0, float(fov_degrees)))
    return (float(image_size_px) / 2.0) / math.tan(math.radians(fov) / 2.0)


def vertical_fov_degrees(hfov_degrees: float, image_width: float, image_height: float) -> float:
    """Campo vertical coerente com pixels quadrados e o campo horizontal."""
    fx = focal_length_px(image_width, hfov_degrees)
    if fx <= 0 or image_height <= 0:
        return hfov_degrees
    return math.degrees(2.0 * math.atan((image_height / 2.0) / fx))


def distance_from_size(real_size_m: float, size_px: float, focal_px: float) -> float:
    """Lei do pinhole: distância = tamanho_real * focal / tamanho_em_pixels."""
    if size_px <= 1.0 or focal_px <= 0.0 or real_size_m <= 0.0:
        return math.inf
    return (real_size_m * focal_px) / size_px


def ground_plane_distance(
    y_bottom: float,
    image_height: float,
    focal_y: float,
    camera_height_m: float,
    pitch_deg: float = 0.0,
) -> float | None:
    """Distância até o contato com o chão, se o pé do objeto está abaixo do horizonte.

    Inclinação positiva significa a câmera olhando para baixo.
    """
    if camera_height_m <= 0.0 or focal_y <= 0.0 or image_height <= 0:
        return None
    horizon = (image_height / 2.0) - focal_y * math.tan(math.radians(pitch_deg))
    if y_bottom <= horizon + 1.0:
        return None
    angle = math.atan((y_bottom - horizon) / focal_y)
    if angle <= 0.02:
        return None
    distance = camera_height_m / math.tan(angle)
    if not math.isfinite(distance) or distance <= 0.0:
        return None
    return distance


def clamp_distance(distance_m: float, low: float = MIN_DISTANCE_M, high: float = MAX_DISTANCE_M) -> float:
    if not math.isfinite(distance_m):
        return high
    return max(low, min(high, distance_m))


def estimate_bbox_distance(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    label: str,
    hfov_degrees: float = 66.0,
    vfov_degrees: float = 0.0,
    camera_height_m: float = 1.2,
    pitch_deg: float = 0.0,
) -> float:
    """Estimativa rápida sempre disponível, em metros.

    Combina a altura/largura típica da classe com o ponto em que a caixa toca
    o chão. Se o objeto está cortado na borda de baixo, o plano do chão não
    entra, porque o contato real não está visível.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    height_px = max(1.0, y2 - y1)
    width_px = max(1.0, x2 - x1)
    real_h, real_w = REAL_SIZE_M.get(label, (1.0, 0.6))
    vfov = vfov_degrees if vfov_degrees and vfov_degrees > 1.0 else vertical_fov_degrees(
        hfov_degrees, image_width, image_height
    )
    focal_y = focal_length_px(image_height, vfov)
    focal_x = focal_length_px(image_width, hfov_degrees)
    by_height = distance_from_size(real_h, height_px, focal_y)
    by_width = distance_from_size(real_w, width_px, focal_x)

    cut_bottom = y2 >= image_height - 2
    cut_top = y1 <= 1
    cut_side = x1 <= 1 or x2 >= image_width - 2
    if (cut_top or cut_bottom) and not cut_side and math.isfinite(by_width):
        by_size = by_width
    elif cut_side and not (cut_top or cut_bottom):
        by_size = by_height
    elif math.isfinite(by_height) and math.isfinite(by_width):
        by_size = 0.75 * by_height + 0.25 * by_width
    else:
        by_size = by_height if math.isfinite(by_height) else by_width

    if cut_bottom:
        return clamp_distance(by_size)
    by_ground = ground_plane_distance(y2, image_height, focal_y, camera_height_m, pitch_deg)
    if by_ground is None:
        return clamp_distance(by_size)
    return clamp_distance(0.6 * by_size + 0.4 * by_ground)


def fit_depth_scale(pairs: list[tuple[float, float]]) -> tuple[str, float] | None:
    """Acha a escala que leva a profundidade relativa para metros.

    Cada par é (profundidade relativa, metros da estimativa por caixa).
    Com dois ou mais objetos, o sinal da correlação decide se o mapa é
    proporcional à distância ou é o inverso (disparidade).
    """
    clean = [
        (rel, meters)
        for rel, meters in pairs
        if rel is not None and meters is not None and rel > 1e-6 and meters > 0 and math.isfinite(meters)
    ]
    if len(clean) < 2:
        return None
    rels = [item[0] for item in clean]
    meters = [item[1] for item in clean]
    mean_r = sum(rels) / len(rels)
    mean_m = sum(meters) / len(meters)
    numer = sum((r - mean_r) * (m - mean_m) for r, m in clean)
    den_r = math.sqrt(sum((r - mean_r) ** 2 for r in rels))
    den_m = math.sqrt(sum((m - mean_m) ** 2 for m in meters))
    if den_r < 1e-9 or den_m < 1e-9:
        return None
    correlation = numer / (den_r * den_m)
    if correlation >= 0.3:
        ratios = sorted(m / r for r, m in clean)
        return ("proportional", ratios[len(ratios) // 2])
    if correlation <= -0.3:
        products = sorted(m * r for r, m in clean)
        return ("inverse", products[len(products) // 2])
    return None


def fuse_distance(
    bbox_m: float,
    relative_depth: float | None,
    calibration: tuple[str, float] | None,
) -> float:
    """Junta a estimativa geométrica com a profundidade monocular já em escala.

    Sem calibração, a geometria permanece. Com calibração, a profundidade
    pesa um pouco mais porque enxerga o relevo dentro da caixa.
    """
    if relative_depth is None or relative_depth <= 1e-6 or calibration is None:
        return clamp_distance(bbox_m)
    mode, scale = calibration
    if mode == "inverse":
        depth_m = scale / relative_depth
    else:
        depth_m = scale * relative_depth
    if not math.isfinite(depth_m) or depth_m <= 0.0 or depth_m > MAX_DISTANCE_M * 2:
        return clamp_distance(bbox_m)
    return clamp_distance(0.45 * bbox_m + 0.55 * depth_m)
