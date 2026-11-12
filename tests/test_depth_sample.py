"""Amostra da profundidade relativa, sem baixar o modelo."""

import numpy as np

from obstaclesense.depth import sample_relative_depth


def test_sample_prefers_the_lower_part_of_the_box():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[70:, :] = 9
    value = sample_relative_depth(depth, (10, 0, 90, 100))
    assert value == 9


def test_empty_box_returns_nothing():
    depth = np.ones((20, 20), dtype=np.float32)
    assert sample_relative_depth(depth, (0, 0, 0, 0)) is None
