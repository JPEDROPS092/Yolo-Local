"""Faixa central do caminho e lados."""

from obstaclesense.direction import box_center_x, classify_direction


def test_bands_with_margin():
    width = 100
    margin = 0.15
    assert classify_direction(34, width, margin) == "left"
    assert classify_direction(35, width, margin) == "center"
    assert classify_direction(50, width, margin) == "center"
    assert classify_direction(65, width, margin) == "center"
    assert classify_direction(66, width, margin) == "right"


def test_box_center_drives_direction():
    box = (10, 0, 30, 40)
    assert box_center_x(box) == 20
    assert classify_direction(box_center_x(box), 100, 0.15) == "left"


def test_zero_width_stays_center():
    assert classify_direction(0, 0, 0.15) == "center"
