"""Estimativa de distância por geometria e fusão com profundidade relativa."""

import math

from obstaclesense.distance import (
    clamp_distance,
    distance_from_size,
    estimate_bbox_distance,
    fit_depth_scale,
    focal_length_px,
    fuse_distance,
    ground_plane_distance,
    vertical_fov_degrees,
)


def test_focal_length_of_ninety_degrees():
    assert math.isclose(focal_length_px(640, 90), 320.0, rel_tol=1e-6)


def test_pinhole_distance_inverts_size():
    focal = 500
    assert distance_from_size(1.7, 1.7 * focal / 5.0, focal) == 5.0
    far = distance_from_size(1.7, 80, focal)
    near = distance_from_size(1.7, 200, focal)
    assert near < far


def test_ground_plane_matches_camera_height():
    focal_y = 400
    image_height = 480
    pitch = 0
    angle = math.radians(30)
    y_bottom = image_height / 2 + focal_y * math.tan(angle)
    distance = ground_plane_distance(y_bottom, image_height, focal_y, camera_height_m=1.2, pitch_deg=pitch)
    assert distance is not None
    assert abs(distance - (1.2 / math.tan(angle))) < 1e-6


def test_ground_plane_ignores_points_above_horizon():
    assert ground_plane_distance(10, 480, 400, 1.2, pitch_deg=0) is None


def test_larger_box_is_closer():
    far = estimate_bbox_distance((200, 180, 280, 260), 640, 480, "person", hfov_degrees=66, camera_height_m=1.2)
    near = estimate_bbox_distance((180, 40, 460, 470), 640, 480, "person", hfov_degrees=66, camera_height_m=1.2)
    assert near < far
    assert 0.3 <= near <= 30
    assert 0.3 <= far <= 30


def test_known_width_when_feet_are_cut_off():
    # Pessoa de 0,50 m de largura, focal horizontal de 90° em 640 px → focal 320.
    # Caixa cortada embaixo: a largura manda. 0,50 * 320 / 100 = 1,6 m.
    distance = estimate_bbox_distance(
        (270, 10, 370, 479),
        640,
        480,
        "person",
        hfov_degrees=90,
        vfov_degrees=60,
        camera_height_m=1.2,
    )
    assert abs(distance - 1.6) < 0.05


def test_vertical_fov_matches_square_pixels():
    vfov = vertical_fov_degrees(90, 640, 480)
    assert abs(focal_length_px(480, vfov) - focal_length_px(640, 90)) < 1e-6


def test_depth_scale_proportional_and_inverse():
    assert fit_depth_scale([(1, 2), (2, 4), (4, 8)]) == ("proportional", 2)
    mode, scale = fit_depth_scale([(1, 10), (2, 5), (4, 2.5)])
    assert mode == "inverse"
    assert abs(scale - 10) < 1e-6
    assert fit_depth_scale([(1, 3), (2, 8), (3, 2)]) is None
    assert fit_depth_scale([(2, 4)]) is None


def test_fuse_uses_bbox_without_calibration():
    assert fuse_distance(4.0, 2.0, None) == 4.0
    assert fuse_distance(4.0, None, ("proportional", 2)) == 4.0
    mixed = fuse_distance(4.0, 2.0, ("proportional", 1.5))
    assert abs(mixed - (0.45 * 4.0 + 0.55 * 3.0)) < 1e-6
    inverse = fuse_distance(4.0, 2.0, ("inverse", 8.0))
    assert abs(inverse - (0.45 * 4.0 + 0.55 * 4.0)) < 1e-6


def test_clamp_rejects_infinity():
    assert clamp_distance(math.inf) == 30.0
    assert clamp_distance(0.05) == 0.3
