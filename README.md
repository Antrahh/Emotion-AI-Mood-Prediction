# Emotion-Based Mood Prediction using CNN

## Project Overview

This project aims to detect human emotions using a Convolutional Neural Network (CNN) and real-time webcam input.
The system analyzes facial expressions and predicts emotions such as **happy, sad, angry, and neutral**.

The project uses a combination of a **custom dataset** and an **online dataset** to train the model. Images are preprocessed and resized to 48×48 grayscale format to ensure efficient training of the CNN model.

---

## Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. CNN Model Architecture
4. Model Training
5. Real-Time Emotion Detection using Webcam
6. Displaying Predicted Emotion

---

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* CNN (Convolutional Neural Network)

---

## Dataset Details

The dataset used in this project is a combination of:

* **Custom collected images**
* **Online facial emotion dataset**

All images were converted to **grayscale** and resized to **48×48 pixels** for compatibility with the CNN model.

Dataset categories:

* Happy
* Sad
* Angry
* Neutral


##Team Members and Responsibilities

Antra Singh (Team Leader)

Dataset collection and combination
Image preprocessing (grayscale conversion and resizing)
Frontend development
Preprocessing pipeline

Khusbhu

Dataset collection and combination
Model training implementation (training.py)
Frontend development


Samiksha Rawat

Dataset collection and combination
Model training implementation (training.py)
Frontend development


## Project Structure
Emotion-AI-Mood-Prediction

dataset/
    happy/
    sad/
    angry/
    neutral/

preprocess.py
training.py
cnn.py
README.md

## Future Work

* Model training completion
* Real-time webcam emotion detection
* Performance optimization and accuracy improvement

---

## Expected Outcome

The system will be able to detect human emotions from facial expressions in real time using a trained CNN model and display the predicted mood.

