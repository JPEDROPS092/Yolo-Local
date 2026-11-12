"""O mesmo objeto, mais perto a cada quadro, é marcado como aproximação."""

from obstaclesense.models import Obstacle
from obstaclesense.tracking import SimpleTracker, iou


def _box_obstacle(label, distance, box):
    return Obstacle(label, 0.8, distance, "center", "low", box)


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_approaching_flag_and_stable_id():
    tracker = SimpleTracker(approach_drop_m=0.35)
    first = tracker.update([_box_obstacle("person", 5.0, (100, 50, 180, 300))], now=0.0)
    second = tracker.update([_box_obstacle("person", 4.4, (98, 48, 184, 310))], now=0.4)
    third = tracker.update([_box_obstacle("person", 3.8, (96, 40, 190, 330))], now=0.8)
    assert first[0].approaching is False
    assert second[0].track_id == first[0].track_id
    assert third[0].track_id == first[0].track_id
    assert third[0].approaching is True


def test_static_object_is_not_approaching():
    tracker = SimpleTracker(approach_drop_m=0.35)
    box = (100, 50, 180, 300)
    for index in range(4):
        seen = tracker.update([_box_obstacle("chair", 3.0, box)], now=index * 0.3)
    assert seen[0].approaching is False


def test_distant_new_object_gets_another_id():
    tracker = SimpleTracker()
    first = tracker.update([_box_obstacle("person", 4.0, (10, 10, 40, 80))], now=0.0)
    second = tracker.update(
        [
            _box_obstacle("person", 4.0, (10, 10, 40, 80)),
            _box_obstacle("car", 6.0, (400, 200, 500, 320)),
        ],
        now=0.2,
    )
    ids = {item.track_id for item in second}
    assert first[0].track_id in ids
    assert len(ids) == 2
