import cv2
import numpy as np


# Region strengths
REGION_STRENGTH = {
    "left_eye": 0.18,
    "right_eye": 0.18,
    "left_eyebrow": 0.22,
    "right_eyebrow": 0.22,
    "nose": 0.16,
    "mouth": 0.15,
}


def dct_frequency_perturbation(region, strength):

    region = region.astype(np.float32)

    channels = []

    for c in range(3):

        channel = region[:, :, c]

        dct = cv2.dct(channel)

        h, w = dct.shape

        # Skip low-frequency area
        y1 = h // 4
        x1 = w // 4

        # Modify only higher frequencies
        high = dct[y1:, x1:]

        noise = np.random.normal(
            0,
            strength,
            high.shape
        ).astype(np.float32)

        high *= (1.0 + noise)

        dct[y1:, x1:] = high

        channels.append(cv2.idct(dct))

    result = cv2.merge(channels)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_frequency_identity(image, region_masks):

    protected = image.copy()

    for name, mask in region_masks.items():

        if name not in REGION_STRENGTH:
            continue

        roi = cv2.bitwise_and(
            protected,
            protected,
            mask=mask
        )

        transformed = dct_frequency_perturbation(
            roi,
            REGION_STRENGTH[name]
        )

        alpha = cv2.GaussianBlur(
            mask,
            (31, 31),
            10
        ).astype(np.float32) / 255.0

        alpha = alpha[..., None]

        protected = (
            protected.astype(np.float32) * (1 - alpha)
            +
            transformed.astype(np.float32) * alpha
        )

        protected = np.clip(
            protected,
            0,
            255
        ).astype(np.uint8)

    return protected