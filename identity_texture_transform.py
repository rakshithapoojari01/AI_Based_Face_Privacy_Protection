import cv2
import numpy as np

# Region-specific perturbation strengths
REGION_STRENGTH = {
    "left_eye": 0.22,
    "right_eye": 0.22,
    "left_eyebrow": 0.18,
    "right_eyebrow": 0.18,
    "nose": 0.08,
    "mouth": 0.12,
}

def bilateral_texture(image):
    """
    Smooth flat skin while preserving edges.
    """
    return cv2.bilateralFilter(
        image,
        d=7,
        sigmaColor=30,
        sigmaSpace=30
    )

def local_contrast(image, clip=2.0, grid=(8, 8)):
    """
    Improve local contrast without changing colors.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=grid
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

def high_frequency_boost(image, amount=0.25):
    """
    Enhance fine facial texture.
    """

    blur = cv2.GaussianBlur(
        image,
        (0, 0),
        2
    )

    detail = cv2.subtract(
        image,
        blur
    )

    return cv2.addWeighted(
        image,
        1.0,
        detail,
        amount,
        0
    )

def micro_texture_noise(image, strength):
    """
    Add tiny random texture perturbations.
    """

    noise = np.random.normal(
        0,
        strength * 2,
        image.shape
    ).astype(np.float32)

    result = image.astype(np.float32)

    result += noise

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

def brightness_match(original, transformed):
    """
    Preserve overall brightness after texture modification.
    """

    original_lab = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2LAB
    )

    transformed_lab = cv2.cvtColor(
        transformed,
        cv2.COLOR_BGR2LAB
    )

    l1, a1, b1 = cv2.split(original_lab)
    l2, a2, b2 = cv2.split(transformed_lab)

    diff = np.mean(l1) - np.mean(l2)

    l2 = l2.astype(np.float32)

    l2 += diff

    l2 = np.clip(
        l2,
        0,
        255
    ).astype(np.uint8)

    merged = cv2.merge((l2, a2, b2))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

def adaptive_texture_transform(region, strength):
    """
    Adaptive texture transformation for one facial region.
    """

    # Keep a copy of the original ROI
    original = region.copy()

    # 1. Preserve edges while smoothing flat skin
    region = bilateral_texture(region)

    # 2. Improve local contrast
    region = local_contrast(region)

    # 3. Enhance micro texture
    region = high_frequency_boost(region)

    # 4. Add tiny random perturbations
    region = micro_texture_noise(
        region,
        strength
    )

    # 5. Restore brightness
    region = brightness_match(
        original,
        region
    )

    return region

def process_region(image, mask, strength):
    """
    Apply adaptive texture transformation to a single facial region.
    """

    result = image.copy()

    # Find bounding box of the mask
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return result

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Extract ROI
    roi = image[y1:y2+1, x1:x2+1]
    roi_mask = mask[y1:y2+1, x1:x2+1]

    # Transform ROI
    transformed = adaptive_texture_transform(
        roi,
        strength
    )

    # Soft alpha mask
    alpha = cv2.GaussianBlur(
        roi_mask,
        (41, 41),
        15
    ).astype(np.float32) / 255.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Laplacian(gray, cv2.CV_32F)

    edges = np.abs(edges)

    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    edges /= (edges.max() + 1e-6)

    # Make edge-aware alpha (keep it 2D)
    alpha *= (0.4 + 0.6 * edges)

# Convert alpha to 3 channels for BGR blending
    alpha = alpha[..., None]

# Blend
    blended = (
        roi.astype(np.float32) * (1.0 - alpha)
        +
        transformed.astype(np.float32) * alpha
    )
    
    result[y1:y2+1, x1:x2+1] = np.clip(
        blended,
        0,
        255
    ).astype(np.uint8)

    return result

def apply_region_texture(image, region_masks):
    """
    Apply adaptive texture transformation to all identity regions.
    """

    result = image.copy()

    # Left eye
    result = process_region(
        result,
        region_masks["left_eye"],
        REGION_STRENGTH["left_eye"]
    )

    # Right eye
    result = process_region(
        result,
        region_masks["right_eye"],
        REGION_STRENGTH["right_eye"]
    )

    # Left eyebrow
    result = process_region(
        result,
        region_masks["left_eyebrow"],
        REGION_STRENGTH["left_eyebrow"]
    )

    # Right eyebrow
    result = process_region(
        result,
        region_masks["right_eyebrow"],
        REGION_STRENGTH["right_eyebrow"]
    )

    # Nose
    result = process_region(
        result,
        region_masks["nose"],
        REGION_STRENGTH["nose"]
    )

    # Mouth
    result = process_region(
        result,
        region_masks["mouth"],
        REGION_STRENGTH["mouth"]
    )

    return result

def apply_identity_texture(image, mask):
    """
    Apply adaptive texture transformation using a full-face mask.
    This function is kept for compatibility with the rest of the project.
    """

    # Estimate a reasonable strength from the mask coverage
    strength = 0.25

    # Transform the entire image
    transformed = adaptive_texture_transform(
        image,
        strength
    )

    # Create a soft alpha mask
    alpha = cv2.GaussianBlur(
        mask,
        (41, 41),
        15
    ).astype(np.float32) / 255.0

    alpha *= strength
    alpha = alpha[..., None]

    # Blend transformed image with original
    blended = (
        image.astype(np.float32) * (1 - alpha)
        +
        transformed.astype(np.float32) * alpha
    )

    return np.clip(
        blended,
        0,
        255
    ).astype(np.uint8)

