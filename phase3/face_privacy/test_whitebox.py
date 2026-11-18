"""
test_whitebox.py

Run the PURE adversarial pipeline (NO landmark warp -- face_protection.py
is never called) on one image and print the result: similarity, privacy
score, elapsed time, and save the output image for visual inspection.

Usage:
    python test_whitebox.py path\\to\\image.jpg
    python test_whitebox.py path\\to\\image.jpg --epsilon 2 --alpha 0.4 --steps 80
"""

import sys
import time
import argparse

import cv2

from gradient_adversarial import refine_only_whitebox
from face_similarity import get_embedding, cosine_similarity
from privacy_score import calculate_privacy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read image: {args.image}")
        sys.exit(1)

    print(f"Running PURE adversarial (no landmark warp): "
          f"epsilon={args.epsilon}, alpha={args.alpha}, steps={args.steps}")

    start = time.time()
    final_image, mp_result, refine_result = refine_only_whitebox(
        image, epsilon=args.epsilon, alpha=args.alpha,
        num_steps=args.steps, verbose=True,
    )
    elapsed = time.time() - start

    if refine_result["no_face_detected"]:
        print("No face detected during refinement -- nothing more to report.")
        sys.exit(1)

    emb_orig = get_embedding(image)
    emb_final = get_embedding(final_image)
    if emb_orig is not None and emb_final is not None:
        final_similarity = float(cosine_similarity(emb_orig, emb_final))
        score = calculate_privacy_score(final_similarity)
        print()
        print(f"Full-image similarity: {final_similarity:.4f}")
        print(f"Privacy score: {score:.2f}")
    else:
        print("Warning: could not compute full-image similarity.")

    print(f"Elapsed time: {elapsed:.2f}s")

    out_path = "whitebox_test_output.jpg"
    cv2.imwrite(out_path, final_image)
    print(f"Saved output image to: {out_path}")


if __name__ == "__main__":
    main()