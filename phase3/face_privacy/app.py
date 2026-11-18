import os
import time

import cv2
from flask import Flask, render_template, request, redirect, url_for

from gradient_adversarial import refine_only_whitebox
from face_similarity import get_embedding, cosine_similarity
from privacy_score import calculate_privacy_score, get_protection_level

# --------------------------------------------------------------------------
# Config -- adjust these three if your folder names differ
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
PROTECTED_FOLDER = os.path.join(BASE_DIR, "static", "protected")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROTECTED_FOLDER, exist_ok=True)

# Locked-in production parameters -- update these once tune_epsilon.py
# tells you the actual highest-epsilon value that still looks clean.
# These are placeholders until that sweep is run and read.
EPSILON = 4.0
ALPHA = 0.8
NUM_STEPS = 80

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


# <-- REPLACE THE OLD process() FUNCTION HERE

@app.route("/process", methods=["POST"])
def process():

    print("Step 1")

    file = request.files.get("image")
    print("Step 2")

    if not file or file.filename == "":
        print("No file")
        return redirect(url_for("upload_page"))

    filename = file.filename
    print("Step 3")

    original_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(original_path)
    print("Saved:", original_path)

    image = cv2.imread(original_path)
    print("Image loaded:", image is not None)

    print("Starting refine_only_whitebox()")

    final_image, mp_result, refine_result = refine_only_whitebox(
        image,
        epsilon=EPSILON,
        alpha=ALPHA,
        num_steps=NUM_STEPS,
        verbose=False,
    )

    print("Finished refine_only_whitebox()")


if __name__ == "__main__":
    app.run(debug=True)
