import cv2
import numpy as np

from privacy_controller import get_privacy_settings


def elastic_identity_transform(
    image,
    region_masks,
    privacy_mode="balanced",
):
    """
    Apply adaptive elastic deformation based on
    selected privacy mode.
    """

    settings = get_privacy_settings(privacy_mode)

    # Disable elastic transform for Natural mode
    if not settings["enable_elastic"]:
        return image.copy()

    strength = settings["elastic_strength"]

    result = image.copy()

    h, w = image.shape[:2]

    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)

    rng = np.random.default_rng(42)

    # -----------------------------
    # Adaptive deformation settings
    # -----------------------------

    noise_sigma = 1.0 + strength * 4.0

    blur_sigma = 10 + strength * 20

    deformation_scale = 0.5 + strength * 2.5

    for mask in region_masks.values():

        noise_x = rng.normal(
            0,
            noise_sigma,
            (h, w)
        ).astype(np.float32)

        noise_y = rng.normal(
            0,
            noise_sigma,
            (h, w)
        ).astype(np.float32)

        noise_x = cv2.GaussianBlur(
            noise_x,
            (0, 0),
            blur_sigma
        )

        noise_y = cv2.GaussianBlur(
            noise_y,
            (0, 0),
            blur_sigma
        )

        alpha = mask.astype(np.float32) / 255.0

        dx += noise_x * alpha * deformation_scale
        dy += noise_y * alpha * deformation_scale

    x, y = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    result = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return result