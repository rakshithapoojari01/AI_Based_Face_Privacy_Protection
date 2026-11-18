import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load InsightFace model.
# allowed_modules restricts FaceAnalysis to only what this project
# actually uses (face detection + recognition embedding). Without this,
# every call to app.get() also runs 3D landmarks, 2D landmarks, and
# gender/age models that are never read anywhere in this codebase --
# pure wasted inference time on every single request.
app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1)

def get_embedding(image):

    faces = app.get(image)

    if len(faces) == 0:
        return None

    return faces[0].embedding


def cosine_similarity(vec1, vec2):

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) *
        np.linalg.norm(vec2)
    )


def compare_faces(original, protected):

    emb1 = get_embedding(original)
    emb2 = get_embedding(protected)

    if emb1 is None or emb2 is None:
        return None

    similarity = cosine_similarity(
        emb1,
        emb2
    )

    return round(float(similarity), 4)
