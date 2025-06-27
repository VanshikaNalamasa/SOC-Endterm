# SOC-Midterm Report Facial Recognition and Emotional Analysis
> **Author:** Vanshika Nalamasa  
> **Program:** IIT Bombay Summer of Code 2025  
> **Mentors:** [Yash Choudhary, Sahil Kukreja]

---

## 📌 Project Overview

This project explores **recognizing handwritten code or digits using Convolutional Neural Networks (CNNs)**.  
The final goal is to build a complete deep learning pipeline capable of identifying and interpreting handwritten programming code.  
We're starting by working with digit recognition (MNIST), then moving to face recognition and beyond.

---

## 📅 Week-wise Progress

### ✅ Week 1: Fundamentals  
**Topics Covered:**  
- Python: [Kaggle Python Course](https://www.kaggle.com/learn/python)  
- NumPy: [Beginner Tutorial](https://www.kaggle.com/code/legendadnan/numpy-tutorial-for-beginners-data-science)  
- Pandas: [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas)  
- Data Visualization: [Kaggle Data Viz Course](https://www.kaggle.com/learn/data-visualization)  
- Intro to PyTorch: [FreeCodeCamp PyTorch Course](https://www.youtube.com/watch?v=GIsg-ZUy0MY)

**Activities:**  
Completed foundational exercises on Python, NumPy, Pandas, and visualization using interactive Kaggle notebooks.  
Also watched the PyTorch crash course to understand tensor operations and neural network basics.

---

### ✅ Week 2: Deep Learning Theory  
**Topics Covered:**  
- [Neural Networks](https://youtube.com/playlist?list=PLuhqtP7jdD8CftMk831qdE8BlIteSaNzD)  
- [Convolutional Neural Networks](https://youtube.com/playlist?list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu)  
- [Recurrent Neural Networks](https://youtube.com/playlist?list=PLuhqtP7jdD8ARBnzj8SZwNFhwWT89fAFr)

**Activities:**  
Learned the core ideas behind feedforward networks, CNNs, and RNNs.  
Started understanding how CNNs work on image data, which is critical for handwritten digit/code recognition.

---

### ✅ Week 3: Assignment 1 – Handwritten Digit Recognition  

**Objective:**  
Build a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.

**Tools Used:**  
- TensorFlow / Keras  
- CNN with Conv2D, MaxPooling, BatchNormalization  
- Matplotlib for performance visualization

**Result:**  
Achieved a **test accuracy of over 98%**.  
The model was able to learn and generalize well using standard CNN layers.

**📁 File:** [`assignment/handwritten_digit_cnn.py`](assignment/handwritten_digit_cnn.py)

---

### ✅ Week 4: Face Recognition Strategy Study

**Reading Material:**  
[Face Recognition Strategy Paper – ScholarWorks](https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1643&context=etd_projects)

**Activities:**  
- Understood the high-level face recognition pipeline involving:  
  - Face detection  
  - Feature extraction (e.g., embeddings using CNNs)  
  - Classification (e.g., k-NN, SVM)  
- Connected concepts back to digit/code recognition (image → embedding → classification)

---

## 🔍 What's Next (Week 5+)  
- Start building a custom dataset with handwritten code snippets  
- Train a CNN-based model to detect multiple lines of code  
- Explore OpenCV for image pre-processing  
- Consider moving toward sequence recognition (RNNs or CRNNs) for multi-character decoding

---

## 📎 References  
- [PyTorch Crash Course](https://www.youtube.com/watch?v=GIsg-ZUy0MY)  
- [Kaggle Python Course](https://www.kaggle.com/learn/python)  
- [Face Recognition Article](https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1643&context=etd_projects)

---

## 🧾 To Do  
- [ ] Start working on real handwritten code input (Week 5)  
- [ ] Build training pipeline  
- [ ] Optimize model performance  


