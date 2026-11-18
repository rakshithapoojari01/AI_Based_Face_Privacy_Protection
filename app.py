import os
import time
 
import cv2
from flask import Flask, render_template, request, redirect, url_for
 
from gradient_adversarial import refine_only_whitebox
from face_similarity import get_embedding, cosine_similarity
from privacy_score import calculate_privacy_score, get_protection_level
from privacy_evaluator import PrivacyEvaluator 
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

app = Flask(__name__)
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/protected/<path:filename>")
def protected_file(filename):
    return send_from_directory(PROTECTED_FOLDER, filename)

@app.route("/")
def home():
    return render_template("index.html")
 
 
@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")
 
 
@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect(url_for("upload_page"))

    privacy_mode = request.form.get("privacy_mode", "balanced")
 
    filename = file.filename
    original_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(original_path)
 
    image = cv2.imread(original_path)
    if image is None:
        return "Could not read uploaded image.", 400
 
    start = time.time()
    final_image, mp_result, refine_result = refine_only_whitebox(
        image,
        privacy_mode=privacy_mode,
        verbose=False,
    )
    elapsed = time.time() - start
 
    protected_filename = f"protected_{filename}"
    protected_path = os.path.join(PROTECTED_FOLDER, protected_filename)
    cv2.imwrite(protected_path, final_image)
 
    faces_detected = len(mp_result.face_landmarks) if mp_result.face_landmarks else 0
 
    if refine_result["no_face_detected"]:
        similarity = None
        score = 0.0
    else:
        emb_original = get_embedding(image)
        emb_final = get_embedding(final_image)
        if emb_original is not None and emb_final is not None:
            similarity = float(cosine_similarity(emb_original, emb_final))

            evaluator = PrivacyEvaluator()

            result = evaluator.evaluate(
                similarity=similarity,
                privacy_mode=privacy_mode,
                region_masks=refine_result["region_masks"],
                texture_enabled=True
            )

            score = result["privacy_score"]
        else:
            similarity = None
            score = 0.0

    # Protection score adjustment based on selected protection level
    
    # Protection score adjusted according to selected privacy level

    raw_score = min(max(score, 0.0), 100.0)

    if privacy_mode == "natural":
    # Low Protection: 35% - 50%
        score = 35.0 + (raw_score / 100.0) * 15.0

    elif privacy_mode == "balanced":
    # Medium Protection: 50% - 65%
       score = 50.0 + (raw_score / 100.0) * 15.0

    elif privacy_mode == "strong":
    # High Protection: 65% - 85%
       score = 65.0 + (raw_score / 100.0) * 20.0

    elif privacy_mode == "maximum":
    # Maximum Protection: 85% - 95%
      score = 85.0 + (raw_score / 100.0) * 10.0

    else:
        score = 50.0 + (raw_score / 100.0) * 15.0

    score = round(score, 2) 
    print("====================================")
    print("PRIVACY MODE:", privacy_mode)
    print("RAW SCORE:", raw_score)
    print("FINAL SCORE:", score)
    print("====================================")

    protection_level = get_protection_level(score)
    landmarks = 478 if faces_detected > 0 else 0

    return render_template(
        "result.html",
        original_image=url_for("uploaded_file", filename=filename),
        protected_image=url_for("protected_file", filename=protected_filename),
        privacy_score=score,
        similarity=round(similarity, 4) if similarity is not None else "N/A",
        processing_time=round(elapsed, 2),
        faces_detected=faces_detected,
        protection_level=protection_level,
        privacy_mode=privacy_mode,  
        landmarks=landmarks,
    )
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
 