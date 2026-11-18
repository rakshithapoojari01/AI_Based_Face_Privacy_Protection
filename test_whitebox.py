import sys
import time
import argparse
 
import cv2
 
from gradient_adversarial import protect_and_refine_whitebox
from privacy_score import calculate_privacy_score, get_protection_level
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
 
    from gradient_adversarial import protect_and_refine_whitebox

    print(f"Running Stage-1 + Stage-2 protection:"
          f" strength=1.0, epsilon={args.epsilon}, alpha={args.alpha}, steps={args.steps}")

    start = time.time()
    final_image, mp_result, refine_result = protect_and_refine_whitebox(
        image,
        strength=1.0,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.steps,
        verbose=True,
    )
    elapsed = time.time() - start
 
    if refine_result["no_face_detected"]:
        print("No face detected during refinement -- nothing more to report.")
        sys.exit(1)
 
    final_similarity = refine_result["similarity"]

    if final_similarity is not None:
        score = calculate_privacy_score(final_similarity)

        print()
        print(f"Final similarity: {final_similarity:.4f}")
        print(f"Privacy score: {score:.2f}")
        print(f"Protection level: {get_protection_level(score)}")
    else:
        print("Warning: could not compute full-image similarity.")
 
    print(f"Elapsed time: {elapsed:.2f}s")
 
    out_path = "whitebox_test_output.jpg"
    cv2.imwrite(out_path, final_image)
    print(f"Saved output image to: {out_path}")
 
 
if __name__ == "__main__":
    main()

 