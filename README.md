# 🛡️ FaceShield AI
### AI-Based Face Privacy Protection System

**Protecting Facial Identity Through AI-Powered Selective Transformation**

## 📌 About the Project

FaceShield AI is an AI-powered facial privacy protection system designed to reduce the exposure of personal identity in facial images while maintaining their natural visual appearance.

The system uses facial landmark detection and selective identity transformation to identify and modify identity-sensitive facial regions rather than applying ordinary filters to the entire image.

## 🎯 Project Objectives

- Protect individuals against unauthorized facial recognition.
- Identify identity-sensitive facial regions.
- Apply selective facial identity transformation.
- Preserve the natural appearance and usability of images.
- Evaluate the effectiveness of facial privacy protection.

## ✨ Key Features

- Facial landmark detection using MediaPipe.
- Identification of identity-sensitive facial regions.
- Selective identity transformation.
- Face embedding extraction and similarity evaluation.
- Cosine similarity-based identity comparison.
- Privacy score calculation.
- Multiple privacy protection levels.
- Original and protected image comparison.
- Protected image download.

## ⚙️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Backend | Flask |
| Computer Vision | OpenCV, MediaPipe |
| Face Recognition | InsightFace |
| Deep Learning | PyTorch, TensorFlow |
| Numerical Processing | NumPy |
| Frontend | HTML, CSS, JavaScript |

## 🔄 System Workflow

1. **Image Upload:** User uploads a facial image.
2. **Face Detection:** The system detects the face.
3. **Landmark Extraction:** Facial landmarks are extracted.
4. **Sensitive Region Identification:** Identity-sensitive facial regions are identified.
5. **Selective Transformation:** The selected regions undergo identity transformation.
6. **Similarity Evaluation:** Original and protected face embeddings are compared using cosine similarity.
7. **Privacy Evaluation:** A privacy score is calculated.
8. **Result Display:** The protected image and evaluation results are displayed.

## 🧠 Core AI/ML Concepts

- Facial Landmark Detection
- Face Embeddings
- Selective Identity Transformation
- Cosine Similarity
- Computer Vision
- Adversarial Privacy Protection (include only if retained in the current implementation)

## 🖥️ Application Screenshots

Add screenshots of:
- Home page
- Image upload page
- Privacy level selection
- Original versus protected image
- Privacy score result page

## 🚀 Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/rakshithapoojari01/AI_Based_Face_Privacy_Protection.git
```

**2. Navigate to the project**

```bash
cd AI_Based_Face_Privacy_Protection
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the application**

```bash
python app.py
```

Open your browser and visit:

`http://127.0.0.1:5000`

## 🔮 Future Enhancements

- Improve privacy protection performance.
- Enhance image quality preservation.
- Evaluate against additional face recognition models.
- Improve real-time facial privacy protection.
- Explore deployment options.

## 👩‍💻 Project Information

**Project:** FaceShield AI – AI-Based Face Privacy Protection

**Domain:** Artificial Intelligence and Machine Learning

**Repository:** [GitHub – FaceShield AI](https://github.com/rakshithapoojari01/AI_Based_Face_Privacy_Protection)

---

*Developed as an AI/ML project focused on facial identity privacy and responsible AI.*