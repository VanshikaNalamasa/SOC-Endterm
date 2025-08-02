# SOC 2025 – Facial Recognition and Emotion Analysis Project(ID:122)

> **Mentee:** Vanshika Nalamasa  
> **Mentors:** [Yash Choudhary, Sahil Kukreja]  
> **Repo Purpose:** Midterm progress report + working code

--
This project is part of my 7-week Summer of Code journey focused on computer vision and deep learning. The objective was to build a facial expression recognition model using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset. The project involved theory, hands-on implementation, research, and evaluation — culminating in a fully functional expression classifier.

---

## 🧠 Problem Statement

Facial expressions are a vital form of non-verbal communication. Automating their recognition has applications in human-computer interaction, mental health diagnostics, surveillance, and more. In this project, the goal was to classify faces into **7 emotion categories** using deep learning.

---

## 📦 Dataset Used

The [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) includes:
- ~35,000 grayscale 48×48 facial images
- Emotion-labeled images: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Provided as a CSV file with pixel strings and labels

---

## 📅 Weekly Progress Overview

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

### ✅ Week 5: Preparing for Emotion Recognition
- Researched FER-2013 dataset structure and limitations
- Implemented data loading, normalization, and one-hot encoding
- Split data into train/val/test sets
- Visualized class distributions and sample faces

### ✅ Week 6: Model Design & Training
- Designed a CNN architecture with 3 convolutional blocks
- Added BatchNormalization and Dropout for regularization
- Applied data augmentation using ImageDataGenerator
- Used EarlyStopping and ReduceLROnPlateau to avoid overfitting
- Trained the model on 80% of the data with ~60–65% test accuracy

### ✅ Week 7: Evaluation, Visualization & Final Report
- Generated confusion matrix and classification report
- Performed per-class accuracy analysis
- Visualized training history with accuracy and loss plots
- Summarized key findings, limitations, and future work
- Prepared video explanation and this final GitHub README

---

## 🛠️ Implementation Pipeline

### 🧼 Data Preprocessing
- Parsed pixel strings into 48×48 images
- Normalized pixel values to [0, 1]
- One-hot encoded emotion labels for softmax

### ✂️ Data Splitting
- 80% training, 10% validation, 10% testing using stratified splits

### 🔁 Data Augmentation
- Horizontal flips, rotations, shifts, brightness changes
- Implemented using Keras `ImageDataGenerator`

### 🧱 CNN Model Architecture
```python
Conv2D → BatchNorm → Conv2D → MaxPooling → Dropout
Conv2D → BatchNorm → Conv2D → MaxPooling → Dropout
Conv2D → BatchNorm → Conv2D → MaxPooling → Dropout
Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Dense(7, softmax)

⚙️ Training Setup
Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

Callbacks: EarlyStopping + ReduceLROnPlateau

Trained for 30–50 epochs with batch size 32

Results
Metric	Value (Approx.)
Training Acc	↑ steadily (~70–85%)
Validation Acc	~55–65%
Test Accuracy	~60–65%
Best Classes	Happy, Neutral
Hard Classes	Disgust, Fear

 Key Learnings
Data Augmentation significantly boosts generalization.

Even small CNNs can learn meaningful features from low-res images.

BatchNorm + Dropout are effective in stabilizing training.

Class imbalance affects model performance and needs addressing.

uture Scope
Use transfer learning with MobileNet, ResNet, or EfficientNet

Implement weighted loss functions or focal loss

Deploy using OpenCV for real-time webcam emotion detection

Add Grad-CAM to visualize attention areas in the face

Explore RNNs or LSTMs for facial expression sequences in videos

Acknowledgements
Dataset: FER-2013 via Kaggle

Code written in Python using TensorFlow and Keras

References: SJSU paper, YouTube lecture series on CNNs & RNNs

Thanks to project mentors and peers for guidance & support!
 Summary
Over 7 weeks, I went from basic image classification to building a real-world facial expression recognition system. This project brought together theory, coding, experimentation, and interpretation — giving me practical confidence in applying deep learning to vision problems.

yaml
Copy
Edit




