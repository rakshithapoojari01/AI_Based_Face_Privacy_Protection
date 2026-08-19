import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from texture_protection import apply_texture_protection
 
from identity_region_v3 import (
    LEFT_EYE,
    RIGHT_EYE,
    NOSE,
    MOUTH,
    LEFT_EYEBROW,
    RIGHT_EYEBROW,
)
from mesh_generator import generate_triangles
from mesh_warp import warp_triangle
from identity_region_v3 import build_region_masks

from identity_texture_transform import apply_region_texture
from semantic_identity_transform import semantic_landmark_transform 
from privacy_controller import get_privacy_settings
from identity_region_v3 import build_region_masks
from identity_texture_transform import apply_region_texture
from frequency_identity_transform import apply_frequency_identity
# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
 
MAX_FACES = 10
 
# Identity-sensitive regions, grouped so each region can be warped
# coherently (as a group) rather than as independent noisy points.
# Eyebrows are included to widen the disrupted area beyond eyes/nose/
# mouth -- InsightFace's embedding draws on the whole aligned face, so
# a bigger (still natural-looking) changed surface pushes similarity
# down further than warping eyes/nose/mouth alone can.
IDENTITY_REGIONS = {
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "nose": NOSE,
    "mouth": MOUTH,
    "left_eyebrow": LEFT_EYEBROW,
    "right_eyebrow": RIGHT_EYEBROW,
}
 
# Per-region warp strength, expressed as a fraction of the face's
# interocular distance (so the effect scales with face size / image
# resolution instead of being a fixed pixel amount).
REGION_WARP_CONFIG = {

    "left_eye":      (0.09, 8.0, 0.08),
    "right_eye":     (0.09, 8.0, 0.08),

    "nose":          (0.06, 6.0, 0.05),

    "mouth":         (0.10, 8.0, 0.08),

    "left_eyebrow":  (0.05, 3.0, 0.03),
    "right_eyebrow": (0.05, 3.0, 0.03),
}
# How far the smoothing/feathering mask extends past the identity
# landmarks, in pixels, at a "reference" interocular distance of 100px.
# Actual radius is scaled by the detected face size.
FEATHER_MARGIN_RATIO = 0.35
FEATHER_BLUR_KSIZE_RATIO = 0.12  # relative to interocular distance
 
# How far outward (as a ratio of interocular distance) the fixed anchor
# ring is placed around the identity regions. Triangulation is built
# ONLY from identity points + this ring, so every triangle stays small
# and local -- points outside the ring are never touched. This is what
# prevents a single warped triangle from stretching across the whole face.
ANCHOR_RING_MARGIN_RATIO = 0.55
ANCHOR_RING_POINT_COUNT = 32
 
# strength=1.0 maps to REGION_WARP_CONFIG as written above.
STRENGTH_MIN = 0.1
STRENGTH_MAX = 3.0
DEFAULT_STRENGTH = 1.0
 
 
# --------------------------------------------------------------------------
# MediaPipe FaceLandmarker setup  
# --------------------------------------------------------------------------
 
def _create_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"FaceLandmarker model not found at: {MODEL_PATH}. "
            f"Make sure 'face_landmarker.task' is inside the models/ folder."
        )
 
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=MAX_FACES,
    )
    return vision.FaceLandmarker.create_from_options(options)
 
 
detector = _create_detector()
 
 
# --------------------------------------------------------------------------
# Landmark helpers
# --------------------------------------------------------------------------
 
def _landmarks_to_points(face_landmarks, width, height):
    """Convert normalized MediaPipe landmarks to clamped pixel coordinates."""
    points = []
    for lm in face_landmarks:
        x = min(max(int(round(lm.x * width)), 0), width - 1)
        y = min(max(int(round(lm.y * height)), 0), height - 1)
        points.append((x, y))
    return points
 
 
def _interocular_distance(points):
    """Distance between outer left-eye and outer right-eye corners.
 
    Falls back to a safe default if indices are out of range.
    """
    try:
        left_outer = np.array(points[33], dtype=np.float32)
        right_outer = np.array(points[263], dtype=np.float32)
        dist = float(np.linalg.norm(left_outer - right_outer))
        return dist if dist > 1e-3 else 50.0
    except IndexError:
        return 50.0
 
 
def _face_seed(points):
    """Deterministic seed derived from a face's own landmarks.
 
    Ensures the same input image always produces the same protected
    output (important for reproducible evaluation), while different
    faces / different images get different warps.
    """
    arr = np.array(points, dtype=np.int64)
    return int(np.sum(arr[:, 0] * 131 + arr[:, 1] * 977) % (2**31 - 1))
 
 
# --------------------------------------------------------------------------
# Region-level affine transform generation
# --------------------------------------------------------------------------
 
def _random_region_params(rng, max_scale_delta, max_rotation_deg, max_translate_px):
    """Sample one coherent affine transform for a region."""
    scale = 1.0 + rng.uniform(-max_scale_delta, max_scale_delta)
    rotation_rad = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    tx = rng.uniform(-max_translate_px, max_translate_px)
    ty = rng.uniform(-max_translate_px, max_translate_px)
    return scale, rotation_rad, tx, ty
 
 
def _apply_region_transform(points_subset, scale, rotation_rad, tx, ty):
    """Apply a scale + rotation + translation around the region centroid."""
    pts = np.array(points_subset, dtype=np.float32)
    centroid = pts.mean(axis=0)
 
    cos_a, sin_a = np.cos(rotation_rad), np.sin(rotation_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
 
    centered = pts - centroid
    transformed = (centered * scale) @ rot.T
    transformed += centroid
    transformed += np.array([tx, ty], dtype=np.float32)
 
    return transformed

def _apply_landmark_displacement(
    warped_points,
    rng,
    max_disp_px,
):
    """
    Apply small independent displacement to each landmark.
    This changes local facial geometry while preserving
    overall appearance.
    """

    displaced = []

    for x, y in warped_points:

        dx = rng.uniform(-max_disp_px, max_disp_px)
        dy = rng.uniform(-max_disp_px, max_disp_px)

        displaced.append([
            x + dx,
            y + dy
        ])

    return np.array(displaced, dtype=np.float32)
 
 
def _build_target_points(mesh_points, width, height, rng, strength):
    """Return a copy of mesh_points with identity regions warped.
 
    `strength` linearly scales each region's max scale/rotation/translate
    bounds. strength=1.0 uses REGION_WARP_CONFIG as-is.
    """
    new_points = list(mesh_points)
    interocular = _interocular_distance(mesh_points)
 
    for region_name, indices in IDENTITY_REGIONS.items():
        max_scale_delta, max_rotation_deg, max_translate_ratio = REGION_WARP_CONFIG[region_name]
        max_scale_delta *= strength
        max_rotation_deg *= strength
        max_translate_ratio *= strength
        max_translate_px = interocular * max_translate_ratio
 
        scale, rotation_rad, tx, ty = _random_region_params(
            rng, max_scale_delta, max_rotation_deg, max_translate_px
        )
 
        region_points = [mesh_points[i] for i in indices]
        warped = _apply_region_transform(
            region_points,
            scale,
            rotation_rad,
            tx,
            ty,
        )

        max_disp_px = interocular * 0.025

        warped = _apply_landmark_displacement(
           warped,
           rng,
           max_disp_px,
        )
       
        # Keep a 2px inset from the true edge -- cv2.Subdiv2D requires
        # points strictly inside its rectangle, and width-1/height-1
        # can trip that on some OpenCV builds.
        for idx, (wx, wy) in zip(indices, warped):
            clamped_x = min(max(int(round(wx)), 2), width - 3)
            clamped_y = min(max(int(round(wy)), 2), height - 3)
            new_points[idx] = (clamped_x, clamped_y)
 
    return new_points
 
 
# --------------------------------------------------------------------------
# Mesh warping
# --------------------------------------------------------------------------
 
def _generate_anchor_ring(identity_points, width, height, margin_px, num_points):
    """Build a closed ring of fixed (zero-displacement) points around the
    identity regions.
 
    Triangulating identity points TOGETHER WITH this ring (instead of all
    478 face landmarks) guarantees every triangle stays small and local:
    nothing outside the ring can ever be part of a triangle, so a moved
    landmark can never drag a triangle that spans half the face.
    """
    pts = np.array(identity_points, dtype=np.int32)
    hull = cv2.convexHull(pts)
 
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
 
    k = max(int(margin_px), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
    dilated = cv2.dilate(mask, kernel)
 
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
 
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
 
    # Sample the contour at even arc-length steps to get a manageable,
    # evenly-spaced ring rather than every pixel.
    step = max(len(contour) // num_points, 1)
    ring = []
    for i in range(0, len(contour), step):
        x, y = contour[i]
        cx = min(max(int(x), 2), width - 3)
        cy = min(max(int(y), 2), height - 3)
        ring.append((cx, cy))
 
    return ring
 
 
def _warp_identity_regions(protected, mesh_points, new_points, identity_indices, width, height, interocular):
    """Warp identity regions using a triangulation restricted to those
    regions plus a fixed anchor ring, so the warp cannot spread past it.
    """
    identity_indices_list = sorted(identity_indices)
    identity_src = [mesh_points[i] for i in identity_indices_list]
    identity_dst = [new_points[i] for i in identity_indices_list]
 
    margin_px = interocular * ANCHOR_RING_MARGIN_RATIO
    anchor_ring = _generate_anchor_ring(
        identity_src, width, height, margin_px, ANCHOR_RING_POINT_COUNT
    )
 
    if not anchor_ring:
        return False
 
    # Combined local point set: identity points (which move) followed by
    # anchor ring points (which stay fixed). Triangulating this small set
    # -- instead of all 478 landmarks -- keeps every triangle local.
    combined_src = identity_src + anchor_ring
    combined_dst = identity_dst + anchor_ring  # ring target == ring source
 
    # De-duplicate: Subdiv2D errors on inserting the same point twice
    # (can happen if two landmark indices round to the same pixel).
    seen = set()
    dedup_src, dedup_dst = [], []
    for s, d in zip(combined_src, combined_dst):
        if s in seen:
            continue
        seen.add(s)
        dedup_src.append(s)
        dedup_dst.append(d)
 
    triangles = generate_triangles(dedup_src, width, height)
    point_to_index = {pt: idx for idx, pt in enumerate(dedup_src)}
 
    warped_any = False
 
    for tri in triangles:
        idxs = []
        for pt in tri:
            idx = point_to_index.get(pt)
            if idx is None:
                idxs = None
                break
            idxs.append(idx)
 
        # Skip phantom triangles Subdiv2D creates using virtual/border
        # points that don't correspond to a real point in our set.
        if idxs is None:
            continue
 
        src_tri = [dedup_src[i] for i in idxs]
        dst_tri = [dedup_dst[i] for i in idxs]
 
        if src_tri == dst_tri:
            continue
 
        try:
            warp_triangle(protected, src_tri, dst_tri)
            warped_any = True
        except Exception:
            # A degenerate triangle (near-zero area) can occasionally
            # occur at the mesh boundary; skip it rather than fail the
            # whole request.
            continue
 
    return warped_any
 
 
def _feather_blend(original, protected, mesh_points, identity_indices, width, height, interocular):
    """Softly blend the warped output back over the original at the
    boundary of the modified region, to remove any hard triangle edges.
    """
    margin = max(int(interocular * FEATHER_MARGIN_RATIO), 8)
 
    region_pts = np.array([mesh_points[i] for i in identity_indices], dtype=np.int32)
    hull = cv2.convexHull(region_pts)
 
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
 
    dilate_k = max(margin, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    mask = cv2.dilate(mask, kernel)
 
    blur_ksize = int(interocular * FEATHER_BLUR_KSIZE_RATIO)
    blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
    blur_ksize = max(blur_ksize, 5)
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
 
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = (protected.astype(np.float32) * alpha +
               original.astype(np.float32) * (1.0 - alpha))
 
    return blended.astype(np.uint8)
 
 
# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
 
def protect_face(
    image,
    privacy_mode="balanced",
    strength=None
):
    """Apply landmark-based identity suppression to every face in `image`.
 
    Args:
        image: BGR image as loaded by cv2.imread / cv2.VideoCapture.
        strength: Aggressiveness of the identity warp.
            1.0 = default (REGION_WARP_CONFIG as calibrated).
            <1.0 = subtler, more conservative warp, closer to the original.
            >1.0 = stronger warp, larger drop in recognition similarity,
                    at some cost to how close the result looks to the
                    original. Clamped to [STRENGTH_MIN, STRENGTH_MAX].
 
    Returns:
        (protected_image, mediapipe_result)
        protected_image is a new np.ndarray; the original `image` is
        never modified in place.
    """
    print(">>> protect_face CALLED")
    settings = get_privacy_settings(
        privacy_mode
    )

    if strength is None:
    
        strength = settings["warp_strength"]

    if strength < STRENGTH_MIN or strength > STRENGTH_MAX:
        print(f"Warning: strength={strength} out of range "
              f"[{STRENGTH_MIN}, {STRENGTH_MAX}], clamping.")
        strength = min(max(strength, STRENGTH_MIN), STRENGTH_MAX)
 
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
    result = detector.detect(mp_image)
 
    if not result.face_landmarks:
        print("No faces detected.")
        return image.copy(), result
 
    protected = image.copy()
    height, width = protected.shape[:2]
 
    identity_indices = set(
        idx for indices in IDENTITY_REGIONS.values() for idx in indices
    )
 
    print(f"Faces detected: {len(result.face_landmarks)} (strength={strength})")
 
    for face_idx, face_landmarks in enumerate(result.face_landmarks):
        mesh_points = _landmarks_to_points(face_landmarks, width, height)
        interocular = _interocular_distance(mesh_points)
 
        seed = _face_seed(mesh_points)
        rng = np.random.default_rng(seed)
 
        new_points = semantic_landmark_transform(
            mesh_points,
            privacy_mode=privacy_mode,
        )
        warped_any = _warp_identity_regions(
            protected, mesh_points, new_points, identity_indices,
            width, height, interocular
        )
 
        if warped_any:
            protected = _feather_blend(
                image, protected, mesh_points, identity_indices,
                width, height, interocular
            )

            # Build masks for each facial region
            region_masks = build_region_masks(
                protected,
                mesh_points
            )

        # Apply adaptive texture protection
            protected = apply_region_texture(
                protected,
                region_masks
            ) 
        # Apply adaptive region-wise texture transformation
       
        identity_points = [mesh_points[i] for i in identity_indices]

    
       # protected = apply_texture_protection(
        #    protected,
         #   identity_points,
          #  privacy_mode=privacy_mode,
        #)
        print(f"  Face {face_idx + 1}: {len(mesh_points)} landmarks, "
              f"interocular={interocular:.1f}px, warped={warped_any}")
 
    return protected, result
 