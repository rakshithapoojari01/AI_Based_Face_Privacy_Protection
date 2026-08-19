import numpy as np

from privacy_controller import get_privacy_settings

from identity_region_v3 import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EYEBROW,
    RIGHT_EYEBROW,
    NOSE,
    MOUTH
)


def semantic_landmark_transform(
    mesh_points,
    privacy_mode="balanced",
):

    settings = get_privacy_settings(privacy_mode)

    s = settings["warp_strength"]

    new_points = list(mesh_points)

    # ----------------------------------------
    # Natural Appearance
    # ----------------------------------------

    if privacy_mode == "natural":

        eye_dx = 0.6
        eye_dy = 0.3

        eyebrow_dy = -0.2

        nose_scale = 0.995

        mouth_scale = 1.01

    # ----------------------------------------
    # Balanced
    # ----------------------------------------

    elif privacy_mode == "balanced":
 
        eye_dx = 1.6
        eye_dy = 0.8

        eyebrow_dy = -0.5

        nose_scale = 0.985

        mouth_scale = 1.02

    # ----------------------------------------
    # Strong
    # ----------------------------------------

    elif privacy_mode == "strong":

        eye_dx = 2.6
        eye_dy = 1.4

        eyebrow_dy = -0.8

        nose_scale = 0.975

        mouth_scale = 1.04

    # ----------------------------------------
    # Maximum
    # ----------------------------------------

    else:

        eye_dx = 3.8
        eye_dy = 2.0

        eyebrow_dy = -1.2

        nose_scale = 0.965

        mouth_scale = 1.06

    # ---------------- LEFT EYE ----------------

    new_points = move_region(
        new_points,
        LEFT_EYE,
        dx=-eye_dx * s,
        dy=eye_dy * s,
        scale=1.015
    )

    # ---------------- RIGHT EYE ----------------

    new_points = move_region(
        new_points,
        RIGHT_EYE,
        dx=eye_dx * s,
        dy=eye_dy * s,
        scale=1.015
    )

    # ---------------- EYEBROWS ----------------

    new_points = move_region(
        new_points,
        LEFT_EYEBROW,
        dy=eyebrow_dy * s
    )

    new_points = move_region(
        new_points,
        RIGHT_EYEBROW,
        dy=eyebrow_dy * s
    )

    # ---------------- NOSE ----------------

    new_points = move_region(
        new_points,
        NOSE,
        scale=max(nose_scale, 0.97)
    )

    # ---------------- MOUTH ----------------

    new_points = move_region(
        new_points,
        MOUTH,
        scale=min(mouth_scale, 1.05)
    )

    return new_points
def move_region(points, indices, dx=0.0, dy=0.0, scale=1.0):
    
    result = list(points)

    pts = np.array(
        [points[i] for i in indices],
        dtype=np.float32
    )

    center = pts.mean(axis=0)

    vectors = pts - center

    distances = np.linalg.norm(
        vectors,
        axis=1
    )

    max_distance = max(
        distances.max(),
        1e-6
    )

    weights = distances / max_distance

    new_pts = []

    for p, vec, w in zip(
        pts,
        vectors,
        weights
    ):

        # adaptive scaling
        new_p = center + vec * (
            1 + (scale - 1) * w
        )

        # adaptive translation
        new_p[0] += dx * w
        new_p[1] += dy * w

        new_pts.append(new_p)

    for idx, p in zip(indices, new_pts):

        result[idx] = (
            int(round(p[0])),
            int(round(p[1]))
        )

    return result