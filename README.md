# Anime Mirror

Anime Mirror is an AI-powered web application that transforms a user's photo into an anime-style image.

The project combines face detection and image processing with an AnimeGAN-based model to generate an anime-style version of an uploaded image.

## ✨ Features

* Upload an image through the web interface
* Detect facial landmarks using MediaPipe
* Process the uploaded image
* Generate an anime-style result using AnimeGAN
* Display the transformed image in the browser
* Simple web interface

## 🛠️ Technologies

* Python
* HTML
* MediaPipe
* OpenCV
* ONNX Runtime
* AnimeGAN

## 📁 Project Structure

```text
anime-mirror/
├── index.html
├── main.py
├── test_photo.py
├── models/
│   └── face_landmarker.task
├── .gitignore
└── README.md
```

Local datasets, uploaded photos, virtual environments, and model caches are excluded from the repository.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/anime-mirror.git
cd anime-mirror
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the required dependencies

Install the Python packages required by the project.

```bash
pip install mediapipe opencv-python onnxruntime numpy
```

### 4. Run the application

```bash
python main.py
```

Then open the application in your browser.

## 🔒 Privacy

Photos and datasets used during development are not included in this repository. They are excluded using `.gitignore`.

## 📌 Project Purpose

This project was created to explore computer vision, facial landmark detection, image processing, and AI-based image transformation.

## 📄 License

This project is intended for educational and portfolio purposes.
