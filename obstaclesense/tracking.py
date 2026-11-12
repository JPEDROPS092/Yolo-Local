"""Rastreador simples por IoU para marcar objetos que se aproximam."""

from __future__ import annotations

from dataclasses import dataclass, field

from obstaclesense.models import Obstacle


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class _Track:
    track_id: int
    label: str
    box: tuple[int, int, int, int]
    distances: list[tuple[float, float]] = field(default_factory=list)
    missed: int = 0


class SimpleTracker:
    """Associa caixas da mesma classe entre quadros e observa a distância."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        approach_drop_m: float = 0.35,
        history: int = 8,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.approach_drop_m = approach_drop_m
        self.history = history
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(self, obstacles: list[Obstacle], now: float) -> list[Obstacle]:
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for obs_index, obstacle in enumerate(obstacles):
                if track.label != obstacle.label:
                    continue
                overlap = iou(track.box, obstacle.box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_index, obs_index))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_obs: set[int] = set()
        for _overlap, track_index, obs_index in pairs:
            if track_index in used_tracks or obs_index in used_obs:
                continue
            used_tracks.add(track_index)
            used_obs.add(obs_index)
            track = self.tracks[track_index]
            obstacle = obstacles[obs_index]
            track.box = obstacle.box
            track.missed = 0
            track.distances.append((now, obstacle.distance_meters))
            del track.distances[:-self.history]
            obstacle.track_id = track.track_id
            obstacle.approaching = self._is_approaching(track)

        for obs_index, obstacle in enumerate(obstacles):
            if obs_index in used_obs:
                continue
            track = _Track(
                track_id=self._next_id,
                label=obstacle.label,
                box=obstacle.box,
                distances=[(now, obstacle.distance_meters)],
            )
            self._next_id += 1
            self.tracks.append(track)
            obstacle.track_id = track.track_id
            obstacle.approaching = False

        seen = {obstacle.track_id for obstacle in obstacles}
        for track in self.tracks:
            if track.track_id not in seen:
                track.missed += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]
        return obstacles

    def _is_approaching(self, track: _Track) -> bool:
        if len(track.distances) < 3:
            return False
        start_t, start_d = track.distances[0]
        end_t, end_d = track.distances[-1]
        if end_t <= start_t:
            return False
        return (start_d - end_d) >= self.approach_drop_m
