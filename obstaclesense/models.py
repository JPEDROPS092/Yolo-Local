"""Estruturas compartilhadas entre detecção, risco e API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """Caixa crua devolvida pelo detector, antes de distância e risco."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class Obstacle:
    """Obstáculo pronto para a interface, a voz e o contrato do aplicativo."""

    label: str
    confidence: float
    distance_meters: float
    direction: str
    danger_level: str
    box: tuple[int, int, int, int]
    approaching: bool = False
    track_id: int = -1
    danger_score: float = 0.0

    def to_dict(self) -> dict:
        """Campos exigidos pelo aplicativo Android."""
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "distance_meters": round(float(self.distance_meters), 2),
            "direction": self.direction,
            "danger_level": self.danger_level,
        }


@dataclass
class FrameResult:
    """Resultado de um quadro: objetos, frase e tempos."""

    objects: list[Obstacle] = field(default_factory=list)
    guidance: str = "Nenhum obstáculo à frente."
    latency_ms: float = 0.0
    fps: float = 0.0
    annotated: object | None = None

    def to_dict(self) -> dict:
        return {
            "guidance": self.guidance,
            "latency_ms": round(float(self.latency_ms), 1),
            "fps": round(float(self.fps), 1),
            "objects": [item.to_dict() for item in self.objects],
        }
