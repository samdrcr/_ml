# Homework 3: Neural Network Classification System
**Course:** Machine Learning 2026  
**Student Name:** 范權榮 
**Student ID:** 111210557  

---

## 1. Project Overview: DeepSonar OS
This project implements a **Multi-Layer Perceptron (MLP)** neural network designed to classify underwater acoustic signals. The system distinguishes between "Rocks" (cylindrical rock formations) and "Mines" (metal cylinders) based on 60-band sonar frequency data.

The project consists of a **Python Backend** for model training and a **Web-based Frontend** for real-time inference and visualization, optimized for a 100% fullscreen dashboard experience.

### Live Demo
[🔗 View Live DeepSonar Dashboard on GitHub Pages](https://samdrcr.github.io/sonar-deep-learning/)

---

## 2. Technical Architecture

### Model Topology
The neural network was built using **TensorFlow/Keras** with a customized architecture to handle high-dimensional sonar data:

* **Input Layer:** 60 Neurons (matching the 60 sonar frequency bands).
* **Hidden Layer 1:** 24 Neurons with **ReLU** activation.
* **Regularization:** Dropout layer (rate=0.2) to mitigate overfitting and improve generalization.
* **Hidden Layer 2:** 12 Neurons with **ReLU** activation.
* **Output Layer:** 1 Neuron with **Sigmoid** activation for binary probability mapping.



### Optimization Strategy
* **Loss Function:** `binary_crossentropy` (Standard for binary classification).
* **Optimizer:** `Adam` (Adaptive Moment Estimation) for efficient gradient descent.
* **Data Preprocessing:** Features were scaled using `StandardScaler` to ensure mean=0 and variance=1, preventing features with larger scales from dominating the model.

---

## 3. Experimental Results & Comparative Analysis
During development, multiple configurations were tested to find the optimal balance between bias and variance.

### Hyperparameter Comparison Table
| Model Version | Optimizer | Learning Rate | Hidden Layers | Test Accuracy | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1 (Baseline)** | SGD | 0.01 | 1 (12 units) | 64.2% | High bias; model struggled to converge. |
| **V2 (Complex)** | Adam | 0.001 | 3 (64, 32, 16) | 78.5% | Clear overfitting; accuracy dropped on test set. |
| **V3 (DeepSonar)** | **Adam** | **0.001** | **2 (24, 12)** | **87.5%** | **Optimal; Dropout=0.2 stabilized learning.** |

### Activation Function Logic
* **ReLU:** Chosen for hidden layers to avoid the "Vanishing Gradient" problem common in deep networks.
* **Sigmoid:** Essential for the final layer to produce a probability score between $0$ and $1$.

---

## 4. System Implementation & Workflow

### Project Methodology (CRISP-DM)
1.  **Data Acquisition:** Utilized the UCI Sonar dataset (208 samples).
2.  **Feature Engineering:** Applied Z-score normalization to the 60-feature vector.
3.  **Model Iteration:** Adjusted neuron density and dropout rates across 35 epochs.
4.  **Deployment:** Created a browser-based GUI using `index.html` and `ApexCharts` for real-time signal analysis.



### Files in this Repository
* **`sonar_model.py`**: The training engine. Loads data, trains the MLP, and exports weights.
* **`index.html`**: The UI. A 100% viewport dashboard simulating a sonar array interface.
* **`sonar_classifier.h5`**: The trained model state used for inference.

---

## 5. How to Run Locally

### Prerequisites
Install the required machine learning stack via Terminal:
```bash
pip install tensorflow pandas scikit-learn numpy