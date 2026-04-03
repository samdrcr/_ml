# Micro GPT: Character-Level Language Model Implementation

## Personal Information / 個人資料
* **Name:** 范權榮 
* **Student ID:** 111210557
* **Department:** Computer Science and Information Engineering (CSIE)
* **University:** National Quemoy University (NQU)

---

## English Description

### Project Overview
This project implements a "Micro GPT," a scaled-down version of the Generative Pre-trained Transformer (GPT) architecture. The objective is to build a character-level language model that learns the statistical patterns of a provided text dataset (such as names or literary works) and generates new sequences that mimic the input style. 

The implementation focuses on the core mathematical foundations of the Transformer block, specifically the Decoder-only architecture, using the PyTorch framework to handle tensor operations and automatic differentiation.

### Technical Architecture & Logic
The model architecture follows the standard Transformer Decoder bottleneck, which consists of several key stages:

1. **Token & Position Embedding**: 
   Since computers process numbers rather than text, we map each unique character to a high-dimensional vector. Furthermore, because Transformers process data in parallel (unlike RNNs), they have no inherent sense of order. We add "Position Embeddings" to the token vectors to provide the model with information about where each character sits in the sequence.

2. **Multi-Head Self-Attention (The Communication Phase)**: 
   This is the heart of the model. Each character (token) generates three vectors: **Query**, **Key**, and **Value**. 
   * The **Query** asks: "What information am I looking for?"
   * The **Key** says: "This is the information I contain."
   * The **Value** is the actual content.
   By calculating the dot product of Queries and Keys, the model determines how much "attention" to pay to other characters in the sequence to predict the next one. Using multiple "heads" allows the model to learn different types of relationships (e.g., one head for grammar, one for vowels).

3. **Feed-Forward Network (The Computation Phase)**: 
   After the attention mechanism gathers information from the context, the Feed-Forward Network (FFN) processes this information. It consists of two linear layers with a ReLU activation in between, allowing the model to learn complex non-linear patterns.

4. **Residual Connections & Layer Normalization**: 
   To prevent the "vanishing gradient" problem common in deep learning, we use residual connections (adding the input of a layer back to its output). Layer Normalization is applied to keep the data distributions stable throughout the network.

5. **Softmax Output**: 
   The final linear layer projects the hidden states back to the vocabulary size. A Softmax function is applied to turn these scores into probabilities, from which we sample the next character.

### System Requirements
* Python 3.8+
* PyTorch 2.0+
* CUDA-capable GPU (Optional, defaults to CPU)

### How to Run
1. Ensure `torch` is installed: `pip install torch`.
2. Place your training data in a file named `input.txt` in the root directory.
3. Execute the training script: `python micro_gpt.py`.
4. The model will output the training loss every 500 iterations and generate a sample of text upon completion.

---

## 中文說明 (繁體中文)

### 專案概述
本專案實現了一個「Micro GPT」，這是生成式預訓練變換器 (GPT) 架構的精簡版本。本實作的核心目標是建立一個「字元級語言模型」(Character-level Language Model)，透過學習特定數據集的統計規律，進而生成風格一致的新文本序列。

本專案專注於 Transformer 區塊的核心數學基礎，特別是僅解碼器 (Decoder-only) 的架構，並利用 PyTorch 框架進行張量運算與自動微分處理。

### 技術架構與邏輯
模型設計遵循標準的 Transformer Decoder 結構，包含以下關鍵階段：

1. **符號與位置編碼 (Token & Position Embedding)**：
   由於電腦無法直接處理文字，我們將每個不重複的字元映射為高維度向量。此外，Transformer 採用並行處理，本身缺乏順序感，因此我們在字元向量中加入「位置編碼」，讓模型知道每個字元在序列中的相對位置。

2. **多頭自注意力機制 (Multi-Head Self-Attention - 通訊階段)**：
   這是模型的核心。每個字元會生成三個向量：**Query (查詢)**、**Key (鍵)** 與 **Value (值)**。
   * **Query** 詢問：「我在尋找什麼資訊？」
   * **Key** 表示：「這是我所包含的資訊內容。」
   * **Value** 則是實際的數據內容。
   透過計算 Query 與 Key 的點積，模型可以決定預測下一個字元時，應對序列中其他字元投放多少「注意力」。使用「多頭」機制則能讓模型同時學習多種不同的關聯性（例如：一組頭學習語法，另一組學習母音規律）。

3. **前饋網路 (Feed-Forward Network - 計算階段)**：
   當注意力機制收集完上下文資訊後，前饋網路 (FFN) 會對這些資訊進行深度處理。它由兩個線性層與一個 ReLU 激活函數組成，使模型具備學習複雜非線性模式的能力。

4. **殘差連接與層歸一化 (Residual & LayerNorm)**：
   為了防止深度學習中常見的「梯度消失」問題，我們使用了殘差連接。層歸一化則確保數據分佈在網路傳遞過程中保持穩定。

5. **Softmax 輸出層**：
   最後的線性層將隱藏狀態映射回詞彙表大小，並透過 Softmax 函數將數值轉換為機率分佈，從中抽樣產生下一個字元。

### 訓練與效能挑戰
* **過擬合 (Overfitting)**：當數據集較小時，模型容易死記硬背訓練數據。透過調整 Dropout 或模型層數可以緩解此現象。
* **算力消耗**：雖然是「Micro」級別，但在 CPU 上訓練仍需一定時間。本專案支援 CUDA 加速，建議在具備 GPU 的環境下執行。

### 執行方式
1. 確認已安裝 PyTorch：`pip install torch`。
2. 將訓練數據存放於根目錄下的 `input.txt`。
3. 執行訓練程式碼：`python micro_gpt.py`。
4. 程式將每 500 次迭代顯示一次訓練損失 (Loss)，並在訓練結束後自動生成範例文字。