import os
import sys

import cv2
import numpy as np

import cv2
import mediapipe as mp

from face_protection import protect_face
from face_protection import detector
from face_protection import _landmarks_to_points

from identity_region_v3 import build_region_masks
from identity_texture_transform import apply_region_texture
from elastic_identity_transform import elastic_identity_transform
from privacy_controller import get_privacy_settings
from identity_texture_transform import (
    apply_region_texture,
    apply_identity_texture,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PRIVACY_PRESETS = {

    "natural": {
        "texture": 0.15,
        "elastic": 0.10,
    },

    "balanced": {
        "texture": 0.25,
        "elastic": 0.25,
    },

    "strong": {
        "texture": 0.40,
        "elastic": 0.45,
    },

    "maximum": {
        "texture": 0.60,
        "elastic": 0.70,
    },
}

DEFAULT_EPSILON = (
    4.0              # max per-pixel-channel deviation, 0-255 scale -- tight, near-imperceptible
)
DEFAULT_ALPHA = 0.6 # PGD step size per iteration, 0-255 scale -- small steps
DEFAULT_NUM_STEPS = 120 # more steps to compensate for the smaller step size

MASK_DILATE_RATIO = 0.12
MASK_BLUR_KSIZE_RATIO = 0.10


# --------------------------------------------------------------------------
# Lazy imports -- torch/onnx2torch/insightface are only required if this
# module is actually used, so importing gradient_adversarial.py doesn't
# blow up an environment that only uses the black-box path.
# --------------------------------------------------------------------------


def _lazy_imports():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for gradient_adversarial.py. "
            "Install with: pip install torch --break-system-packages"
        ) from e

    try:
        from onnx2torch import convert
    except ImportError as e:
        raise ImportError(
            "onnx2torch is required for gradient_adversarial.py. "
            "Install with: pip install onnx2torch --break-system-packages"
        ) from e

    try:
        from insightface.utils import face_align
    except ImportError as e:
        raise ImportError(
            "insightface is required for gradient_adversarial.py "
            "(it's already a dependency of this project -- if this "
            "fails, check your environment/venv)."
        ) from e

    return torch, convert, face_align


# --------------------------------------------------------------------------
# Recognition model extraction + conversion
# --------------------------------------------------------------------------

_torch_model_cache = {}


def _get_recognition_model(insightface_app):
    """Pull the recognition sub-model out of an InsightFace FaceAnalysis
    app, convert it to a differentiable torch module, and cache it (the
    conversion is the slow part -- only do it once per process).

    Returns:
        (torch_model, input_size, input_mean, input_std)
    """
    torch, convert, _ = _lazy_imports()

    cache_key = id(insightface_app)
    if cache_key in _torch_model_cache:
        return _torch_model_cache[cache_key]

    rec_model = None
    for taskname, model in insightface_app.models.items():
        if taskname == "recognition":
            rec_model = model
            break

    if rec_model is None:
        raise RuntimeError(
            "Could not find a 'recognition' model inside insightface_app.models. "
            f"Available task names: {list(insightface_app.models.keys())}. "
            "This can happen if your insightface version names tasks "
            "differently -- check `app.models.keys()` yourself and adjust "
            "the taskname check above if needed."
        )

    model_path = getattr(rec_model, "model_file", None)
    if model_path is None or not os.path.exists(model_path):
        raise RuntimeError(
            f"Recognition model file not found at: {model_path}. "
            "Could not locate the .onnx file to convert."
        )

    input_size = getattr(rec_model, "input_size", (112, 112))
    input_mean = getattr(rec_model, "input_mean", 127.5)
    input_std = getattr(rec_model, "input_std", 127.5)

    print(f"Converting recognition model to torch: {model_path}")
    torch_model = convert(model_path)
    torch_model.eval()
    for p in torch_model.parameters():
        p.requires_grad_(False)

    result = (torch_model, input_size, input_mean, input_std)
    _torch_model_cache[cache_key] = result
    return result


def _preprocess_torch(aligned_bgr, input_mean, input_std, torch):
    """BGR uint8/float HxWx3 aligned crop -> normalized NCHW torch tensor,
    matching InsightFace's own preprocessing (RGB, (x-mean)/std).
    """
    rgb = aligned_bgr[:, :, ::-1]  # BGR -> RGB
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float()
    tensor = (tensor - input_mean) / input_std
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW, add batch dim
    return tensor


def _cosine_similarity_torch(a, b, torch):
    a = a.flatten()
    b = b.flatten()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)


# --------------------------------------------------------------------------
# Alignment helpers
# --------------------------------------------------------------------------


def _get_alignment_matrix(insightface_app, image, face_align):
    """Detect a face in `image` and return its 2x3 alignment matrix
    (maps original image -> canonical 112x112 crop), using InsightFace's
    own detector + 5-point landmarks so this matches what the
    recognition model was actually trained/calibrated on.
    """
    faces = insightface_app.get(image)
    if not faces:
        return None
    kps = faces[0].kps  # 5x2: left eye, right eye, nose, mouth-left, mouth-right
    M = face_align.estimate_norm(kps, image_size=112, mode="arcface")
    return M


# --------------------------------------------------------------------------
# Mask (identity regions), reused pattern from adversarial_refine.py
# --------------------------------------------------------------------------


def _build_soft_mask_from_mediapipe(image, width, height):
    """Build the identity-region soft mask using this project's existing
    MediaPipe-based landmark detector, independent of InsightFace's
    5-point kps (which are too coarse for region masking).
    """
    from face_protection import (
        detector,
        IDENTITY_REGIONS,
        _landmarks_to_points,
        _interocular_distance,
    )
    import mediapipe as mp

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None

    mesh_points = _landmarks_to_points(result.face_landmarks[0], width, height)
    interocular = _interocular_distance(mesh_points)
    identity_indices = set(
        idx for indices in IDENTITY_REGIONS.values() for idx in indices
    )

    region_pts = np.array([mesh_points[i] for i in identity_indices], dtype=np.int32)
    hull = cv2.convexHull(region_pts)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    dilate_px = max(int(interocular * MASK_DILATE_RATIO), 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
    )
    mask = cv2.dilate(mask, kernel)

    blur_ksize = int(interocular * MASK_BLUR_KSIZE_RATIO)
    blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
    blur_ksize = max(blur_ksize, 5)
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    return mask.astype(np.float32) / 255.0


def _compute_texture_weight(image_bgr, floor=0.12):
    """Per-pixel perturbation budget multiplier in [floor, 1.0], based on
    local image texture (standard deviation in a small window).

    Human vision is far more sensitive to noise on flat, uniform areas
    (skin) than on textured ones (hair, eyebrows, eye corners), even at
    identical pixel-level magnitude -- that's what caused the visible
    blotches on forehead/cheek. This weight map lets the PGD budget
    concentrate where existing texture can actually mask it, instead of
    spreading evenly and showing up worst exactly on the smoothest skin.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    mean_sq = cv2.blur(gray * gray, (7, 7))
    variance = np.clip(mean_sq - mean * mean, 0, None)
    std = np.sqrt(variance)

    max_std = std.max()
    if max_std < 1e-6:
        # Degenerate (totally flat) image -- don't perturb at all rather
        # than divide by ~zero and treat everything as "high texture".
        return np.full(gray.shape, floor, dtype=np.float32)

    weight = std / max_std
    weight = floor + (1.0 - floor) * weight
    return weight.astype(np.float32)


# --------------------------------------------------------------------------
# Core PGD attack
# --------------------------------------------------------------------------


def refine_identity_suppression_whitebox(
    insightface_app,
    original_image,
    protected_image,
    epsilon=DEFAULT_EPSILON,
    alpha=DEFAULT_ALPHA,
    num_steps=DEFAULT_NUM_STEPS,
    verbose=True,
):
    """Gradient-based (PGD) adversarial refinement on top of an
    already landmark-warped `protected_image`.

    Args:
        insightface_app: the insightface.app.FaceAnalysis instance from
            face_similarity.py (already loaded with buffalo_s).
        original_image, protected_image: BGR np.ndarray.
        epsilon: L-infinity budget, 0-255 scale, around the Stage-1
            protected image (in the aligned 112x112 space).
        alpha: PGD step size, 0-255 scale.
        num_steps: number of gradient steps.

    Returns:
        dict: image, similarity (final measured, in aligned space),
        no_face_detected, history (per-step similarity).
    """
    torch, convert, face_align = _lazy_imports()

    height, width = protected_image.shape[:2]
    

    M = _get_alignment_matrix(insightface_app, original_image, face_align)
    if M is None:
        if verbose:
            print(
                "Refinement skipped: InsightFace found no face in the original image."
            )
        return {
            "image": protected_image.copy(),
            "similarity": None,
            "history": [],
            "no_face_detected": True,
        }

    mask_full = _build_soft_mask_from_mediapipe(protected_image, width, height)
    if mask_full is None:
        if verbose:
            print("Refinement skipped: MediaPipe found no face in the protected image.")
        return {
            "image": protected_image.copy(),
            "similarity": None,
            "history": [],
            "no_face_detected": True,
        }

    torch_model, input_size, input_mean, input_std = _get_recognition_model(
        insightface_app
    )

    orig_aligned = cv2.warpAffine(original_image, M, (112, 112), borderValue=0.0)
    protected_aligned = cv2.warpAffine(protected_image, M, (112, 112), borderValue=0.0)
    # The mask, warped into the same aligned space, restricts the PGD
    # perturbation to identity regions even inside the 112x112 crop.
    mask_aligned = cv2.warpAffine(mask_full, M, (112, 112), borderValue=0.0)
    mask_aligned_t = torch.from_numpy(mask_aligned).float().unsqueeze(0)  # 1xHxW

    # Per-pixel epsilon budget: full `epsilon` on textured pixels (hair,
    # eyebrows, eye corners), much less on flat skin (forehead, cheeks),
    # so the same total perturbation "hides" instead of showing up as
    # blotches on the smoothest areas.
    texture_weight = _compute_texture_weight(protected_aligned)
    epsilon_map = (
        torch.from_numpy(epsilon * texture_weight).float().unsqueeze(-1)
    )  # HxWx1

    with torch.no_grad():
        emb_orig = torch_model(
            _preprocess_torch(orig_aligned, input_mean, input_std, torch)
        )
        emb_orig = emb_orig.flatten().detach()

    x0 = torch.from_numpy(
        protected_aligned[:, :, ::-1].astype(np.float32)
    ).clone()

    x = x0.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=alpha)

    history = []

    if verbose:
        print(
            f"White-box PGD start: epsilon={epsilon}, alpha={alpha}, num_steps={num_steps}"
        )

    for step in range(num_steps):
        normalized = (x - input_mean) / input_std
        inp = normalized.permute(2, 0, 1).unsqueeze(0)

        emb_adv = torch_model(inp).flatten()
        similarity = _cosine_similarity_torch(emb_orig, emb_adv, torch)

        tv_loss = (
            torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :])) +
            torch.mean(torch.abs(x[1:, :, :] - x[:-1, :, :]))
        )

        history.append((step, float(similarity.item())))
        if verbose and (step % 5 == 0 or step == num_steps - 1):
            print(f"  step {step}: similarity={similarity.item():.4f}")

        # We want to MINIMIZE similarity, i.e. gradient DESCENT on it.
        
        optimizer.zero_grad()

        # Total Variation Loss
        tv_loss = (
            torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :])) +
            torch.mean(torch.abs(x[1:, :, :] - x[:-1, :, :]))
        )

        # Combined Loss
        loss = similarity + 0.01 * tv_loss

        loss.backward()

        with torch.no_grad():
           x.grad *= mask_aligned_t.permute(1, 2, 0)

        optimizer.step()

        with torch.no_grad():

            x[:] = torch.max(torch.min(x, x0 + epsilon_map), x0 - epsilon_map)

            x[:] = torch.clamp(x, 0, 255)

        x.requires_grad_(True)

    refined_aligned_rgb = x.detach().numpy().astype(np.uint8)
    refined_aligned_bgr = refined_aligned_rgb[:, :, ::-1]

    # Only warp back the PERTURBATION (refined - original-at-this-stage),
    # not the absolute low-res pixels. The aligned crop is a fixed
    # 112x112 -- warping its actual pixel VALUES back up to full image
    # resolution means upsampling a small image into a much bigger
    # region, which loses sharpness and shows up as a soft/blurry patch
    # (that's the visible seam you saw). The perturbation itself is
    # small and smooth by construction (bounded by epsilon), so warping
    # THAT back and adding it onto the untouched full-resolution image
    # preserves all of the original's detail everywhere.
    delta_aligned = refined_aligned_bgr.astype(np.float32) - protected_aligned.astype(
        np.float32
    )

    M_inv = cv2.invertAffineTransform(M)
    delta_full = cv2.warpAffine(delta_aligned, M_inv, (width, height), borderValue=0.0)

    # Only pixels that actually came from inside the aligned crop (and
    # inside the identity mask) should receive any perturbation --
    # everywhere else stays exactly as Stage 1 left it.
    coverage = cv2.warpAffine(
        np.ones((112, 112), dtype=np.float32), M_inv, (width, height), borderValue=0.0
    )
    alpha_full = np.clip(coverage, 0, 1) * mask_full
    alpha_full = alpha_full[..., None]

    if verbose:
        print(f"  DEBUG: delta_aligned max abs = {np.abs(delta_aligned).max():.3f}")
        print(f"  DEBUG: delta_full max abs = {np.abs(delta_full).max():.3f}")
        print(
            f"  DEBUG: coverage max = {coverage.max():.3f}, mean = {coverage.mean():.4f}"
        )
        print(
            f"  DEBUG: mask_full max = {mask_full.max():.3f}, mean = {mask_full.mean():.4f}"
        )
        print(
            f"  DEBUG: alpha_full max = {alpha_full.max():.3f}, mean = {alpha_full.mean():.4f}"
        )
        print(
            f"  DEBUG: (delta_full * alpha_full) max abs = {np.abs(delta_full * alpha_full).max():.3f}"
        )

    refined_full = protected_image.astype(np.float32) + delta_full * alpha_full
    refined_full = np.clip(refined_full, 0, 255).astype(np.uint8)

    if verbose:
        actual_change = np.abs(
            refined_full.astype(np.float32) - protected_image.astype(np.float32)
        )
        print(
            f"  DEBUG: actual saved-image change, max abs = {actual_change.max():.3f}, "
            f"pixels changed (>1) = {(actual_change.max(axis=2) > 1).sum()}"
        )

    final_similarity = history[-1][1] if history else None

    if verbose:
        print(
            f"White-box PGD done: similarity {history[0][1]:.4f} -> {final_similarity:.4f}"
            if history
            else "White-box PGD done: no steps run"
        )

    return {
        "image": refined_full,
        "similarity": final_similarity,
        "history": history,
        "no_face_detected": False,
    }


# --------------------------------------------------------------------------
# Convenience: full pipeline
# --------------------------------------------------------------------------
def refine_only_whitebox(
    image,
    privacy_mode="balanced",
    verbose=True,
):
    """
    Custom AI Face Privacy pipeline.

    Stage 1:
        Landmark-based identity suppression.

    Stage 2:
        Region-aware texture transformation.

    ArcFace is NOT used here.
    It is only used later by app.py for evaluation.
    """
    settings = get_privacy_settings(privacy_mode)
    
    import cv2
    import mediapipe as mp

    from face_protection import protect_face
    from face_protection import detector
    from face_protection import _landmarks_to_points

    from identity_region_v3 import build_region_masks
    from identity_texture_transform import apply_region_texture

    # -------------------------------------------------------
    # Stage 1 : Landmark deformation
    # -------------------------------------------------------

    protected, mp_result = protect_face(
        image,
        privacy_mode=privacy_mode
    )

    if not mp_result.face_landmarks:

        return (
            protected,
            mp_result,
            {
                "image": protected,
                "similarity": None,
                "history": [],
                "no_face_detected": True,
            },
        )

    # -------------------------------------------------------
    # Stage 2 : Region-aware texture transformation
    # -------------------------------------------------------

    height, width = protected.shape[:2]
    region_masks = {}
    
    for face in mp_result.face_landmarks:

        mesh_points = _landmarks_to_points(
            face,
            width,
            height
        )

        region_masks = build_region_masks(
            protected,
            mesh_points
        )

        if settings["enable_elastic"]:
            protected = elastic_identity_transform(
                protected,
                region_masks,
                privacy_mode=privacy_mode,
            )
        if settings["enable_texture"]:
            protected = apply_region_texture(
                protected,
                region_masks,
            )

        if settings["enable_identity_texture"]:
    
           combined_mask = np.zeros((height, width), dtype=np.uint8)

           for mask in region_masks.values():
               combined_mask = cv2.max(combined_mask, mask)

        protected = apply_identity_texture(
            protected,
            combined_mask
        )
        refine_result = {
           "image": protected,
           "similarity": None,
           "history": [],
           "no_face_detected": False,
           "region_masks": region_masks
        }
    print("Custom privacy pipeline is running...")
    return protected, mp_result, refine_result

def protect_and_refine_whitebox(
    image,
    strength=1.0,
    epsilon=DEFAULT_EPSILON,
    alpha=DEFAULT_ALPHA,
    num_steps=DEFAULT_NUM_STEPS,
    verbose=True,
):
    """protect_face() (Stage 1) + gradient PGD refinement (Stage 2) in
    one call.

    Returns:
        (final_image, mp_result, refine_result)
    """
    
    from face_protection import protect_face
    from face_similarity import app as insightface_app

    protected, mp_result = protect_face(
        image
    )

    refine_result = refine_identity_suppression_whitebox(
        insightface_app,
        image,
        protected,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        verbose=verbose,
    )

    return refine_result["image"], mp_result, refine_result


# --------------------------------------------------------------------------
# Diagnostics -- run this FIRST on your machine
# --------------------------------------------------------------------------


def diagnose(image_path):
    """Fast environment + shape sanity check. Run this before trying the
    full attack -- it fails fast with a clear message if something about
    your installed insightface/torch/onnx2torch versions doesn't match
    what this module assumes, instead of failing deep inside a PGD loop.

    Usage: python gradient_adversarial.py path/to/image.jpg
    """
    print("1. Checking imports...")
    torch, convert, face_align = _lazy_imports()
    print("   OK: torch, onnx2torch, insightface.utils.face_align all import.")

    print("2. Loading InsightFace app (face_similarity.app)...")
    from face_similarity import app as insightface_app

    print(f"   OK: models loaded: {list(insightface_app.models.keys())}")

    print("3. Reading test image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"   FAILED: could not read {image_path}")
        return
    print(f"   OK: shape={image.shape}")

    print("4. Detecting face + alignment matrix...")
    M = _get_alignment_matrix(insightface_app, image, face_align)
    if M is None:
        print("   FAILED: no face detected by insightface_app.get(image)")
        return
    print(f"   OK: alignment matrix:\n{M}")

    print(
        "5. Converting recognition model to torch (this can take a bit the first time)..."
    )
    torch_model, input_size, input_mean, input_std = _get_recognition_model(
        insightface_app
    )
    print(
        f"   OK: input_size={input_size}, input_mean={input_mean}, input_std={input_std}"
    )

    print("6. Running one forward + backward pass...")
    aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
    x = (
        torch.from_numpy(aligned[:, :, ::-1].astype(np.float32))
        .clone()
        .requires_grad_(True)
    )
    normalized = (x - input_mean) / input_std
    inp = normalized.permute(2, 0, 1).unsqueeze(0)
    emb = torch_model(inp)
    print(f"   OK: embedding shape={tuple(emb.shape)}")
    loss = emb.sum()
    loss.backward()
    print(
        f"   OK: gradient shape={tuple(x.grad.shape)}, grad is not all-zero: {bool(x.grad.abs().sum() > 0)}"
    )

    print(
        "\nAll checks passed. You should be able to run refine_identity_suppression_whitebox()"
        " or protect_and_refine_whitebox() on this image now."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gradient_adversarial.py path/to/image.jpg")
        sys.exit(1)
    diagnose(sys.argv[1])
