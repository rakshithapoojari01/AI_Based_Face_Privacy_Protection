# Left Eye
LEFT_EYE = [
    33, 7, 163, 144, 145, 153,
    154, 155, 133, 173, 157,
    158, 159, 160, 161, 246
]

# Right Eye
RIGHT_EYE = [
    362, 382, 381, 380, 374,
    373, 390, 249, 263, 466,
    388, 387, 386, 385, 384,
    398
]

# Nose
NOSE = [
    1,2,4,5,6,19,94,97,98,
    168,195,197,327,326,294
]

# Mouth
MOUTH = [
    61,40,39,37,0,267,269,
    270,291,308,415,310,311,
    312,13,14,17,84,181,91
]

# Left Eyebrow
LEFT_EYEBROW = [
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46
]

# Right Eyebrow
RIGHT_EYEBROW = [
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276
]

import cv2
import numpy as np


IDENTITY_POINTS = (
    LEFT_EYE
    + RIGHT_EYE
    + NOSE
    + MOUTH
    + LEFT_EYEBROW
    + RIGHT_EYEBROW
)


def build_identity_mask(image, landmarks):
    """
    landmarks = list of (x,y) pixel coordinates from MediaPipe
    """

    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array(
        [landmarks[i] for i in IDENTITY_POINTS],
        dtype=np.int32
    )

    hull = cv2.convexHull(pts)

    cv2.fillConvexPoly(mask, hull, 255)

    mask = cv2.GaussianBlur(mask, (41, 41), 0)

    return mask

def build_region_masks(image, landmarks):
    """
    Returns individual masks for each identity region.
    """

    h, w = image.shape[:2]

    regions = {
        "left_eye": LEFT_EYE,
        "right_eye": RIGHT_EYE,
        "left_eyebrow": LEFT_EYEBROW,
        "right_eyebrow": RIGHT_EYEBROW,
        "nose": NOSE,
        "mouth": MOUTH,
    }

    masks = {}

    for name, indices in regions.items():

        mask = np.zeros((h, w), dtype=np.uint8)

        pts = np.array(
            [landmarks[i] for i in indices],
            dtype=np.int32
        )

        hull = cv2.convexHull(pts)

        cv2.fillConvexPoly(mask, hull, 255)

        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        masks[name] = mask

    return masks

def compute_strength(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    texture = cv2.Laplacian(gray, cv2.CV_32F).var()

    brightness = np.mean(gray)

    # Normalize
    texture = min(texture / 500.0, 1.0)
    brightness = brightness / 255.0

    strength = (
        0.6 * texture +
        0.4 * (1.0 - brightness)
    )

    return np.clip(strength, 0.2, 1.0)