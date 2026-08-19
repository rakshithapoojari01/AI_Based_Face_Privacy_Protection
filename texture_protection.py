import cv2
import numpy as np
from privacy_controller import get_privacy_settings

def apply_texture_protection(
    image,
    landmarks,
    privacy_mode="balanced",
):
    print(">>> apply_texture_protection CALLED")

    settings = get_privacy_settings(privacy_mode)

    strength = settings["texture_strength"]

    protected = image.copy()

    h, w = protected.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array(landmarks, dtype=np.int32)

    if len(pts) > 2:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    noise_sigma = 10 + 20 * strength

    noise = np.random.normal(
       0,
       noise_sigma,
       protected.shape
    ).astype(np.float32)

    if strength < 0.2:
        kernel = (5,5)

    elif strength < 0.4:
        kernel = (9,9)

    else:
        kernel = (13,13)

    blur = cv2.GaussianBlur(
        noise,
        kernel,
        0
    )

    protected = protected.astype(np.float32)

    protected = protected + blur * strength

    protected = np.clip(
        protected,
        0,
        255
    ).astype(np.uint8)

    mask = cv2.GaussianBlur(mask, (31,31), 0)

    alpha = mask.astype(np.float32)/255.0

    alpha = alpha[:,:,None]

    result = image.astype(np.float32)*(1-alpha)+protected.astype(np.float32)*alpha

    cv2.imwrite("debug_texture.jpg", result)
    
    return result.astype(np.uint8)