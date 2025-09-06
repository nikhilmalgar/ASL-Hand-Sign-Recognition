# ASL Hand Sign Recognition

## Overview

This ASL Hand Sign Recognition project is a deep learning–powered application that detects and classifies American Sign Language (ASL) characters in real time. Built with **PyTorch** for training and **Streamlit** for deployment, the app uses a webcam to capture hand gestures, predicts the corresponding ASL character, and outputs both text and voice feedback for accessibility.

## Usage

When you launch `app.py`, the real-time inference mode starts automatically. The app displays webcam input, predicted ASL characters, and provides voice output of the prediction.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Training](#model-training)
- [App Deployment](#app-deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- 📷 Webcam-based real-time ASL recognition
- 🧠 PyTorch deep learning model trained on custom dataset
- 🎙️ Voice output for accessibility
- 📊 Accuracy graphs & performance metrics
- 🌗 Dark/Light UI modes with a clean Streamlit interface
- 🔤 Supports A–Z ASL character recognition

## Requirements

- Python 3.13.2
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- gTTS (for text-to-speech)

## Installation

**Clone the Repository:**

git clone https://github.com/nikhilmalgar/ASL-Hand-Sign-Recognition.git
cd ASL-Hand-Sign-Recognition

## Install Dependencies:

pip install -r requirements.txt

## Run the Application:

streamlit run app.py

## Model Training

If you’d like to retrain the model on your dataset, follow these steps:

# Data Preparation

- Collect ASL hand sign images (A–Z).

- Organize them into class-wise folders inside dataset/.

## Training

- Open the Jupyter/Colab notebook train_asl_model.ipynb and run all cells.

- Adjust NUM_CLASSES = 26 if modifying classes.

- The trained model is saved as asl_model.pth.

## Exporting

Convert the model for deployment:

torch.save(model.state_dict(), "asl_model.pth")

## App Deployment

The app integrates:

- Webcam Input → Captures real-time hand gestures

- Model Inference → Predicts ASL character

- Text-to-Speech (gTTS) → Reads prediction aloud

## License

This project is licensed under the MIT License: MIT

