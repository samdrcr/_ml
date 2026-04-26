# Character-Level Generative Transformer (Micro-GPT)

## 1. Author Information
- **Student Name:** 范權榮 
- **Student ID:** 111210557
- **Department:** Computer Science and Information Engineering (CSIE) 資工三
- **Institution:** National Quemoy University (NQU) 國立金門大學

---

## 2. Project Description
This project implements a **Generative Pre-trained Transformer (GPT)** scaled down to a character-level model. It is designed to demonstrate the power of the Transformer architecture in capturing complex linguistic patterns from raw text data. 

Unlike traditional Recurrent Neural Networks (RNNs) or LSTMs, this model leverages **Self-Attention mechanisms** to process entire sequences of data simultaneously, allowing for better long-range dependency modeling and parallelization during training.

### How it Works
The model is trained on a "Next Token Prediction" task. Given a sequence of characters, the model must predict the character that most likely follows. Through thousands of iterations of training on a large corpus (such as Shakespearean drama), the model eventually "learns" grammar, punctuation, character dialogue formatting, and even poetic meter.



---

## 3. Technical Architecture
The implementation follows the modular "Decoder-only" architecture used in the original GPT papers. The core components include:

### A. Scaled Dot-Product Attention
The engine of the Transformer. It calculates the "affinity" between different characters in a block of text using three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.
The mathematical formula used is:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### B. Multi-Head Attention Assembly
Instead of performing a single attention function, the model runs multiple "heads" in parallel. This allows the model to attend to different types of information (e.g., one head focuses on punctuation, another on sentence structure) at the same time.



### C. Processing Blocks (Transformer Blocks)
Each block consists of two main sub-layers:
1. **Communication Layer:** Multi-head self-attention with a causal mask (to prevent looking at future characters).
2. **Computation Layer:** A Position-wise Feed-Forward Network (FFN) that processes the gathered information.

### D. Optimization Features
- **Residual Connections:** Adds the input of a layer to its output to prevent vanishing gradients.
- **Layer Normalization:** Normalizes the activations to ensure stable training.
- **Dropout:** Randomly deactivates neurons during training to prevent overfitting.

---

## 4. Repository Structure
- `architecture.py`: Defines the neural network structure, including the Attention, Feed-Forward, and Transformer Block classes.
- `train.py`: Handles data ingestion, character-level tokenization, hyperparameter configuration, and the main training loop.
- `requirements.txt`: Specifies the Python environment dependencies.
- `data.txt`: The raw text file used for training (e.g., *The Complete Works of Shakespeare*).
- `transformer_model.pth`: The saved model weights (generated after training).

---

## 5. Hyperparameter Configuration
The model is highly configurable via `train.py`. The default settings provided are:
- **Batch Size:** 64 (number of independent sequences processed in parallel)
- **Block Size:** 256 (maximum context length for predictions)
- **Embedding Dimensions:** 384
- **Number of Heads:** 6
- **Number of Layers:** 6
- **Learning Rate:** 3e-4 (AdamW Optimizer)

---

## 6. Setup and Execution

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- A CUDA-capable GPU (recommended for speed, though CPU is supported)

### Installation
1. **Clone the project:**
   ```bash
   git clone <your-repository-link>
   cd <repository-folder>