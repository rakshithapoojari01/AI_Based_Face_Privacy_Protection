import os
import time

import cv2
import numpy as np
import streamlit as st

from gradient_adversarial import protect_and_refine_whitebox
from face_similarity import get_embedding, cosine_similarity
from privacy_score import calculate_privacy_score

st.set_page_config(
    page_title="AI Face Privacy Protection",
    layout="wide"
)

st.title("🛡️ AI-Based Face Privacy Protection")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -------------------------
    # Read image
    # -------------------------

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Could not read image.")
        st.stop()

    original = image.copy()

    with st.spinner("Protecting face..."):

        start = time.time()

        protected_image, mp_result, refine_result = protect_and_refine_whitebox(
        image,
        strength=0.5,
        epsilon=8,
        alpha=2,
        num_steps=20,
        verbose=False
)

        processing_time = time.time() - start

        end = time.time()

    # -------------------------
    # Face Detection
    # -------------------------

    if mp_result.face_landmarks:

        st.success("✅ Face Detected Successfully")

    else:

        st.error("❌ No Face Detected")

        st.stop()

    # -------------------------
    # Similarity
    # -------------------------

    emb_original = get_embedding(original)
    emb_protected = get_embedding(protected_image)

    if emb_original is not None and emb_protected is not None:

        similarity = float(
            cosine_similarity(
                emb_original,
                emb_protected
            )
        )

        privacy_score = calculate_privacy_score(
            similarity
        )

    else:

        similarity = 0.0
        privacy_score = 0.0

    # -------------------------
    # Images
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            cv2.cvtColor(
                original,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    with col2:

        st.subheader("Protected Image")

        st.image(
            cv2.cvtColor(
                protected_image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    # -------------------------
    # Save Images
    # -------------------------

    os.makedirs("output", exist_ok=True)

    cv2.imwrite(
        "output/protected_face.jpg",
        protected_image
    )

    comparison = cv2.hconcat(
        [original, protected_image]
    )

    cv2.imwrite(
        "output/comparison.jpg",
        comparison
    )

    # -------------------------
    # Download
    # -------------------------

    success, buffer = cv2.imencode(
        ".jpg",
        protected_image
    )

    st.download_button(
        label="⬇ Download Protected Image",
        data=buffer.tobytes(),
        file_name="protected_face.jpg",
        mime="image/jpeg"
    )

    # -------------------------
    # Metrics
    # -------------------------

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Similarity",
            f"{similarity:.4f}"
        )

    with c2:

        st.metric(
            "Privacy Score",
            f"{privacy_score:.2f}%"
        )

    with c3:

        st.metric(
            "Processing Time",
            f"{end-start:.2f} sec"
        )

    st.progress(int(min(max(privacy_score, 0), 100)))

    # -------------------------
    # Image Details
    # -------------------------

    st.markdown("---")

    st.subheader("Image Details")

    d1, d2 = st.columns(2)

    with d1:

        st.metric(
            "Width",
            f"{original.shape[1]} px"
        )

    with d2:

        st.metric(
            "Height",
            f"{original.shape[0]} px"
        )

    st.markdown("---")

    st.caption(
        "AI-Based Face Privacy Protection using MediaPipe • OpenCV • InsightFace • White-box Adversarial Refinement • Streamlit"
    )