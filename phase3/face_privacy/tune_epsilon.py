"""
tune_epsilon.py

Focused sweep over epsilon (with alpha scaled proportionally) for the
PURE adversarial pipeline (refine_only_whitebox -- no landmark warp),
now that texture-weighting is in place. Finds the highest epsilon that
still looks natural, instead of guessing.

Each run takes ~15-30s (80 steps), so a sweep of 5-6 values takes a
few minutes. Saves every output image so you can visually compare them
side by side, not just trust the numbers.

Usage:
    python tune_epsilon.py path\\to\\image.jpg
    python tune_epsilon.py path\\to\\image.jpg --epsilons 2,3,4,5,6,8 --steps 80
"""

import argparse
import time

import cv2
from skimage.metrics import structural_similarity as ssim

from gradient_adversarial import refine_only_whitebox
from face_similarity import get_embedding, cosine_similarity
from privacy_score import calculate_privacy_score


def _naturalness_score(original, candidate):
    a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(a, b, full=True)
    return float(score)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image")

    parser.add_argument(
        "--epsilons",
        type=str,
        default="2,3,4,5,6,8"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=80
    )

    parser.add_argument(
        "--alpha-ratio",
        type=float,
        default=0.2,
        help="alpha = epsilon * ratio"
    )

    parser.add_argument(
        "--no-gradient-weighting",
        action="store_true",
        help="Disable gradient-guided weighting"
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read image: {args.image}")
        return

    emb_original = get_embedding(image)
    if emb_original is None:
        print("InsightFace found no face in the original image.")
        return

    epsilons = [float(e) for e in args.epsilons.split(",")]

    print(f"{'epsilon':>7} | {'alpha':>6} | {'similarity':>10} | {'score':>6} | {'ssim':>5} | {'time':>6}")
    print("-" * 60)

    for epsilon in epsilons:
        alpha = round(epsilon * args.alpha_ratio, 3)

        start = time.time()
        final_image, mp_result, refine_result = refine_only_whitebox(
            image, epsilon=epsilon, alpha=alpha, num_steps=args.steps,
            use_gradient_weighting=not args.no_gradient_weighting, verbose=False,
        )
        elapsed = time.time() - start

        if refine_result["no_face_detected"]:
            print(f"{epsilon:>7.1f} | {alpha:>6.2f} | {'--':>10} | {'--':>6} | {'--':>5} | {elapsed:>5.1f}s  face lost")
            continue

        emb_final = get_embedding(final_image)
        similarity = float(cosine_similarity(emb_original, emb_final))
        score = calculate_privacy_score(similarity)
        naturalness = _naturalness_score(image, final_image)

        print(f"{epsilon:>7.1f} | {alpha:>6.2f} | {similarity:>10.4f} | {score:>6.2f} | {naturalness:>5.3f} | {elapsed:>5.1f}s")

        out_path = f"epsilon_test_e{epsilon}.jpg"
        cv2.imwrite(out_path, final_image)

    print("\nSaved one image per epsilon value (epsilon_test_e*.jpg) in this folder.")
    print("Open them side by side with the original and pick the highest epsilon")
    print("that still looks natural to you -- SSIM is a helpful signal but your")
    print("own eyes are the real test here, especially for your project demo.")


if __name__ == "__main__":
    main()