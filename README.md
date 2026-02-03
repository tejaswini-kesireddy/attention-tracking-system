# attention-tracking-system

A real-time computer vision system that monitors student attentiveness by analyzing eye gaze and head movements.

---

## Project Overview

This project uses **Python**, **MediaPipe**, **OpenCV**, **TensorFlow**, **NumPy**, and **Matplotlib** to create a system that tracks and classifies student attention in real time. It is designed to help educators gain insights into student engagement during lectures or online sessions.

---

## Getting Started

Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
  pip install opencv-python mediapipe tensorflow numpy matplotlib
```

---

## Instructions

1. [landmarks.py](landmarks/landmarks.py) - Generate facial landmarks that will be used as training data.
2. [attention_train.py](train/attention_train.py) - Trains and generates a custom h5 model artifact using a base keras model.
3. [predictions.py](test/predictions.py) - Runs the trained model and generate predictions saved into a csv file.
4. [plots.py](utils/plots.py) - Generates a histogram based on the predictions.

---
