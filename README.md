# FaceShield AI — AI-Based Face Privacy Protection

FaceShield AI is a web-based facial privacy protection project that uses facial landmark detection and selective identity transformation to help reduce facial identity exposure while preserving the overall appearance of an image.

## Features

- **Face Image Upload:** Upload an image for processing.
- **Facial Landmark Detection:** Analyze facial regions using computer vision.
- **Selective Facial Protection:** Apply transformations to identity-sensitive facial regions.
- **Privacy Evaluation:** Calculate a privacy protection score.
- **Protection Levels:** Choose a protection level based on your needs.
- **Protected Image Preview:** Compare the original and protected images.
- **Download:** Save the resulting protected image.

## Technologies Used

- Python
- Flask
- OpenCV
- MediaPipe Face Landmarker
- InsightFace
- NumPy
- HTML, CSS and JavaScript

## Project Workflow

1. Upload an image.
2. Detect the face and analyze facial landmarks.
3. Apply the selected facial protection transformations.
4. Evaluate the result using face embeddings, cosine similarity and privacy evaluation.
5. Display the protection results and allow the user to download the image.

## Project Structure

```text
AI_Based_Face_Privacy_Protection/
├── static/
├── templates/
├── app.py
├── face_protection.py
├── face_similarity.py
├── elastic_identity_transform.py
├── frequency_identity_transform.py
├── gradient_adversarial.py
├── identity_region_v3.py
├── identity_texture_transform.py
├── privacy_controller.py
├── privacy_evaluator.py
├── privacy_score.py
├── semantic_identity_transform.py
├── texture_protection.py
├── mesh_generator.py
├── mesh_warp.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rakshithapoojari01/AI_Based_Face_Privacy_Protection.git
cd AI_Based_Face_Privacy_Protection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local address displayed in the terminal in your browser.

## Privacy and Limitations

Protection effectiveness may vary depending on the input image, face detection quality and selected protection level. A privacy score is an evaluation metric and does not guarantee complete anonymity or prevent every form of facial recognition.

## Author

**Rakshitha Poojari**

GitHub: [rakshithapoojari01](https://github.com/rakshithapoojari01)
