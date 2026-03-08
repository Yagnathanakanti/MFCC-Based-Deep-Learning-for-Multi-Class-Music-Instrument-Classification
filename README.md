# 🎵 Music Instrument Classification Using Deep Learning

## 📌 Project Overview

This project implements a **Music Instrument Classification System** using **Deep Learning and Convolutional Neural Networks (CNNs)**.
The system automatically analyzes audio recordings and classifies them into different musical instrument categories based on their acoustic characteristics.

The audio signals are first processed to extract meaningful features such as **Mel Frequency Cepstral Coefficients (MFCCs)**. These features capture the spectral properties of sound, which differ across instruments. The extracted features are then fed into a **CNN model** that learns patterns in the audio and performs multi-class classification.

The proposed system achieves **high classification accuracy (~98%)** and demonstrates the effectiveness of deep learning techniques for audio signal analysis.

---

# 🎯 Objectives

* Automatically classify musical instruments from audio recordings.
* Extract meaningful audio features using signal processing techniques.
* Train a deep learning model capable of recognizing patterns in music.
* Evaluate model performance using accuracy, classification reports, and confusion matrices.

---

# 📂 Dataset

The dataset used in this project is publicly available on Kaggle.

**Dataset Link:**
https://www.kaggle.com/datasets/abdulvahap/music-instrunment-sounds-for-classification

### Dataset Description

The dataset contains **audio recordings (.wav files)** of multiple musical instruments. Each instrument has its own folder containing several sound samples.

Example dataset structure:

```
music_dataset/
│
├── piano/
│   ├── audio1.wav
│   ├── audio2.wav
│
├── guitar/
│   ├── audio1.wav
│   ├── audio2.wav
│
├── violin/
├── flute/
├── trumpet/
├── saxophone/
└── drums/
```

Each folder represents a **separate instrument class**.

---

# ⚙️ Technologies Used

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| Python             | Core programming language               |
| TensorFlow / Keras | Deep learning model development         |
| Librosa            | Audio processing and feature extraction |
| NumPy              | Numerical computations                  |
| Matplotlib         | Data visualization                      |
| Seaborn            | Confusion matrix visualization          |
| KaggleHub          | Dataset downloading                     |

---

# 🧠 Methodology

The system follows a **machine learning pipeline** consisting of several stages.

## 1️⃣ Dataset Download

The dataset is downloaded directly from Kaggle using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download(
"abdulvahap/music-instrunment-sounds-for-classification"
)
```

---

## 2️⃣ Audio Loading

Audio files are loaded using **Librosa**, which converts the sound signal into a numerical waveform.

```python
y, sr = librosa.load(file)
```

Where:

* **y** → audio waveform
* **sr** → sampling rate

---

## 3️⃣ Feature Extraction

Raw audio signals are converted into meaningful features using **MFCC (Mel Frequency Cepstral Coefficients)**.

```python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
```

MFCC features capture:

* pitch
* timbre
* frequency distribution

These characteristics vary between musical instruments, making them useful for classification.

---

## 4️⃣ Data Preprocessing

Since audio clips have different durations, MFCC matrices are standardized using **padding or truncation**.

```
Max Padding = 1000 frames
MFCC Shape = (40, 1000)
```

This ensures consistent input dimensions for the neural network.

---

## 5️⃣ Dataset Splitting

The dataset is divided into three parts:

| Dataset        | Percentage | Purpose                    |
| -------------- | ---------- | -------------------------- |
| Training Set   | 60%        | Train the model            |
| Validation Set | 20%        | Tune hyperparameters       |
| Test Set       | 20%        | Evaluate final performance |

---

# 🤖 Model Architecture

A **Convolutional Neural Network (CNN)** is used for classification.

### CNN Layers

1. Convolution Layer – extracts audio patterns
2. ReLU Activation – introduces non-linearity
3. MaxPooling Layer – reduces feature dimensions
4. Dropout – prevents overfitting
5. Fully Connected Layer – performs classification
6. Softmax Layer – outputs probability of each class

Example architecture:

```
Conv2D (32 filters)
ReLU
Conv2D (32 filters)
MaxPooling
Dropout

Conv2D (64 filters)
ReLU
MaxPooling
Dropout

Conv2D (128 filters)
ReLU
MaxPooling
Dropout

Flatten
Dense (128)
Dropout
Dense (Number of Classes)
Softmax
```

---

# 📊 Model Evaluation

The model performance is evaluated using:

### Accuracy

Percentage of correctly classified audio samples.

### Precision

Measures the correctness of positive predictions.

### Recall

Measures the ability of the model to detect all relevant classes.

### F1 Score

Harmonic mean of precision and recall.

### Confusion Matrix

Shows correct vs incorrect predictions for each class.

---

# 📈 Results

The CNN model achieved:

```
Test Accuracy: ~98%
Weighted F1 Score: ~0.98
```

The confusion matrix demonstrates strong classification performance across multiple instrument classes.

---

# 🔍 Example Output

Input:

```
Audio file: violin_sound.wav
```

Model Prediction:

```
Predicted Instrument: Violin
Confidence: 96%
```

---

# 🚀 Applications

This system can be used in:

* Music streaming services
* Automatic playlist generation
* Music recommendation systems
* Audio tagging and organization
* Music information retrieval

---

# 📌 Future Improvements

Possible enhancements include:

* Adding more instrument classes
* Using larger audio datasets
* Implementing transformer-based audio models
* Real-time instrument detection
* Deployment as a web application

---

# 👩‍💻 Author

**Thanakanti Yagna**
Integrated M.Tech – Computer Science and Engineering
VIT Vellore

GitHub: https://github.com/Yagnathanakanti
