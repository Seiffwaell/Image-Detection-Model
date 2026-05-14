
# 🧠 Supervised Learning Project: CIFAR-10 Vision AI

This repository contains a full-stack web application designed to demonstrate the power of various Deep Learning architectures on the **CIFAR-10** image classification dataset.

## ✨ Features
*   **Multiple Neural Network Architectures:** Choose between a Standard CNN, Simple RNN, GRU, or a highly advanced **LSTM with a Custom Attention Layer**.
*   **Sequence Model Adaptation:** The backend intelligently handles sequence-based models (RNN, GRU, LSTM) by taking a single uploaded image and duplicating it across 3 timesteps to create a static "video sequence" for the models to process.
*   **Sleek Modern GUI:** Built from scratch using Vanilla HTML/CSS/JS and Flask, featuring a fully responsive, dark-mode glassmorphism design with drag-and-drop functionality.
*   **Real-time Inference:** View immediate predictions and dynamic confidence probability bars generated securely in the backend using TensorFlow/Keras.

## 🛠️ Technology Stack
*   **Backend:** Python, Flask, Werkzeug, Pillow
*   **Machine Learning:** TensorFlow, Keras, NumPy
*   **Frontend:** Vanilla JavaScript, HTML5, CSS3

## 🚀 How to use
Simply drag and drop an image of an airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck into the upload zone, select the AI model you wish to test, and click **Analyze Image**!
