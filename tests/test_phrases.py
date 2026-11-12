"""Frases só em português, no formato pedido para a navegação."""

import re

from obstaclesense.phrases import (
    PhraseObstacle,
    build_guidance,
    distance_words,
)

ENGLISH = {
    "person",
    "people",
    "car",
    "cars",
    "bicycle",
    "bike",
    "left",
    "right",
    "center",
    "centre",
    "ahead",
    "warning",
    "danger",
    "high",
    "medium",
    "low",
    "meters",
    "meter",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "front",
    "stop",
    "please",
    "there",
    "is",
    "obstacle",
    "approximately",
    "motorcycle",
    "train",
    "cat",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-zÀ-ÿ]+", text.lower()))


def test_example_phrase_for_a_person_ahead():
    phrase = build_guidance(
        [PhraseObstacle("person", 2.0, "center", "high", approaching=False)]
    )
    assert phrase == (
        "Existe uma pessoa aproximadamente dois metros à sua frente. Desvie para a direita."
    )
    assert not (_words(phrase) & ENGLISH)


def test_distance_words():
    assert distance_words(2.0) == "dois metros"
    assert distance_words(1.0) == "um metro"
    assert distance_words(2.5) == "dois metros e meio"
    assert distance_words(0.4) == "menos de um metro"


def test_vehicle_and_animal_names_are_portuguese():
    bus = build_guidance([PhraseObstacle("bus", 3.0, "left", "high")])
    truck = build_guidance([PhraseObstacle("truck", 4.0, "right", "medium")])
    dog = build_guidance([PhraseObstacle("dog", 1.5, "center", "high")])
    assert "ônibus" in bus
    assert "Desvie para a direita." in bus
    assert "caminhão" in truck
    assert "Desvie para a esquerda." in truck
    assert "cachorro" in dog
    for phrase in (bus, truck, dog):
        assert not (_words(phrase) & ENGLISH)


def test_nearby_person_blocks_the_escape_side():
    phrase = build_guidance(
        [
            PhraseObstacle("bus", 3.4, "center", "medium"),
            PhraseObstacle("person", 2.0, "left", "medium"),
            PhraseObstacle("person", 2.1, "right", "low"),
        ]
    )
    assert phrase.endswith("Pare.")
    assert "ônibus" in phrase


def test_both_sides_blocked_says_stop():
    phrase = build_guidance(
        [
            PhraseObstacle("person", 1.5, "center", "high"),
            PhraseObstacle("car", 2.0, "left", "high"),
            PhraseObstacle("bicycle", 2.0, "right", "medium"),
        ]
    )
    assert phrase.endswith("Pare.")
    assert "pessoa" in phrase


def test_low_danger_does_not_order_a_detour():
    phrase = build_guidance([PhraseObstacle("chair", 8.0, "left", "low")])
    assert phrase == "Existe uma cadeira aproximadamente oito metros à sua esquerda."
    assert "Desvie" not in phrase


def test_approaching_and_very_close():
    approaching = build_guidance([PhraseObstacle("car", 3.0, "center", "high", approaching=True)])
    close = build_guidance([PhraseObstacle("person", 0.4, "center", "high")])
    assert "se aproximando" in approaching
    assert approaching.startswith("Atenção!")
    assert close.startswith("Cuidado!")
    assert "muito perto" in close
    assert not (_words(approaching) & ENGLISH)
    assert not (_words(close) & ENGLISH)


def test_empty_scene():
    assert build_guidance([]) == "Nenhum obstáculo à frente."


def test_unknown_class_does_not_speak_english_label():
    phrase = build_guidance([PhraseObstacle("skateboard", 2.0, "center", "medium")])
    assert "obstáculo" in phrase
    assert "skateboard" not in phrase.lower()
