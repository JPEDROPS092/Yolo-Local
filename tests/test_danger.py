"""Perigo: veículos e o centro do caminho pesam mais que um objeto parado ao lado."""

from obstaclesense.danger import assess_danger, sort_obstacles
from obstaclesense.models import Obstacle


def _obstacle(label, distance, direction, level):
    return Obstacle(label, 0.9, distance, direction, level, (0, 0, 10, 10))


def test_person_two_meters_ahead_is_high():
    level, score = assess_danger(2.0, "center", "person", proximity_m=2.0, collision_threshold=0.7)
    assert level == "high"
    assert score >= 0.7


def test_car_is_more_dangerous_than_person_at_the_same_spot():
    _, person = assess_danger(3.0, "center", "person")
    _, car = assess_danger(3.0, "center", "car")
    assert car > person


def test_close_vehicle_off_to_the_side_is_still_high():
    level, score = assess_danger(2.0, "left", "car", proximity_m=2.0, collision_threshold=0.7)
    assert level == "high"
    assert score >= 0.7


def test_far_chair_is_low():
    level, score = assess_danger(8.0, "left", "chair")
    assert level == "low"
    assert score < 0.7


def test_approaching_raises_the_score():
    _, still = assess_danger(5.0, "center", "car", approaching=False)
    level, moving = assess_danger(5.0, "center", "car", approaching=True)
    assert moving > still
    assert level == "high"


def test_threshold_can_demote_a_warning():
    level, score = assess_danger(2.0, "center", "person", collision_threshold=0.99)
    assert score < 0.99
    assert level == "medium"


def test_same_level_prefers_the_higher_score():
    person = _obstacle("person", 1.9, "left", "medium")
    person.danger_score = 0.52
    bus = _obstacle("bus", 3.4, "center", "medium")
    bus.danger_score = 0.62
    ordered = sort_obstacles([person, bus])
    assert [item.label for item in ordered] == ["bus", "person"]


def test_sort_puts_the_dangerous_one_first():
    ordered = sort_obstacles(
        [
            _obstacle("chair", 1.0, "left", "low"),
            _obstacle("car", 4.0, "center", "high"),
            _obstacle("person", 1.2, "center", "medium"),
        ]
    )
    assert [item.label for item in ordered] == ["car", "person", "chair"]
