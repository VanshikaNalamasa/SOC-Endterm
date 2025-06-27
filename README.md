# SoC 2025 – Facial Recognition and Emotion Analysis Project

> **Author:** Vanshika Nalamasa  
> **Program:**  Seasons of Code 2025  
> **Mentors:** [Yash Choudhary, Sahil Kukreja]  
> **Repo Purpose:** Midterm progress report + working code

---

##  Project Overview

This project involves building a system that can:
1. **Recognize individuals** from facial features (face recognition)
2. **Identify emotional states** (emotion analysis: happy, sad, angry, etc.)

The system uses deep learning (mainly CNNs, possibly RNNs) to process images or videos and output identity and emotion.  
Later stages will involve multi-task learning and real-time performance tuning using tools like **Grad-CAM**, **heatmaps**, and **quantization**.

---

## 📅 Week-wise Progress

###  Week 1: Foundations – Python & Libraries  
**Topics:**  
- [Python - Kaggle](https://www.kaggle.com/learn/python)  
- [NumPy](https://www.kaggle.com/code/legendadnan/numpy-tutorial-for-beginners-data-science)  
- [Pandas](https://www.kaggle.com/learn/pandas)  
- [Data Visualization](https://www.kaggle.com/learn/data-visualization)  
- [PyTorch Crash Course](https://www.youtube.com/watch?v=GIsg-ZUy0MY)

**Work Done:**  
Practiced Python libraries and data handling via Kaggle exercises.  
Watched the freeCodeCamp PyTorch tutorial to understand tensors, layers, and training loops.

---

###  Week 2: Deep Learning Theory – Core Concepts  
**Topics:**  
- [Neural Networks Playlist](https://youtube.com/playlist?list=PLuhqtP7jdD8CftMk831qdE8BlIteSaNzD)  
- [CNNs Playlist](https://youtube.com/playlist?list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu)  
- [RNNs Playlist](https://youtube.com/playlist?list=PLuhqtP7jdD8ARBnzj8SZwNFhwWT89fAFr)

**Work Done:**  
Studied theoretical foundations of feedforward networks, CNNs, and RNNs.  
Focused on convolution layers, pooling, and emotion-sequence modeling with RNNs.

---

###  Week 3: Assignment – Handwritten Digit Recognition

**Task:**  
Build a CNN model using the MNIST dataset to recognize handwritten digits (0–9).  

**File:** [`assignment/handwritten_digit_cnn.py`](assignment/handwritten_digit_cnn.py)

**Skills Practiced:**  
- Preprocessing image data  
- CNN modeling using Conv2D, MaxPooling2D, Dropout  
- Evaluating using train/val/test split  
- Visualization of accuracy and loss over epochs

**Result:**  
Achieved **~98% test accuracy** – solid hands-on for CNNs ahead of face recognition.

---

###  Week 4: Understanding Face Recognition Strategy

**Reading:**  
📄 [Face Recognition Strategy & Implementation Paper – SJSU](https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1643&context=etd_projects)

**Insights Gained:**  
- Face detection, cropping, and normalization pipeline  
- Feature extraction using CNNs  
- Emotion analysis using multi-head or multitask learning  
- Visualization using **Grad-CAM** and heatmaps  
- Real-time image stream handling + deployment considerations

---

## 🔍 What’s Next (Weeks 5+)  

- Begin collecting small-scale face + emotion dataset  
- Build initial face detection and preprocessing pipeline  
- Fine-tune CNN on emotion-labeled face images  
- Integrate Grad-CAM for visual interpretability  
- Explore RNNs if working with temporal (video) data

---

## 📁 Folder Structure



