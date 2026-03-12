# Lab 1: Simple Linear Regression / 實驗室 1：單變數線性迴歸

**Name / 姓名：** 范權榮  
**Student ID / 學號：** 111210557  

---

## 1. Objective / 實驗目標
The goal of this lab is to manually implement a **Simple Linear Regression** model using **Gradient Descent**. Instead of relying on high-level libraries like Scikit-Learn, this implementation focuses on the mathematical optimization process to find the best-fit line for a given dataset.

本實驗的主要目標是透過 **梯度下降法 (Gradient Descent)** 手動實作一個簡單的線性迴歸模型。我們不直接調用 Scikit-Learn 等現成庫，而是透過數學優化過程來找出數據的最佳擬合線。



## 2. Methodology / 實作方法
The implementation follows these core concepts: / 本次實作包含以下核心概念：

* **Cost Function (MSE) / 損失函數**: We use Mean Squared Error to measure the variance between predicted values and actual data points. / 使用均方誤差來衡量預測值與實際值之間的差距。
* **Gradient Calculation / 梯度計算**: We calculate the partial derivatives for both the slope ($m$) and the intercept ($c$). / 計算斜率 $m$ 與截距 $c$ 的偏微分。
    $$D_m = \frac{-2}{n} \sum_{i=1}^{n} x_i (y_i - y_{pred})$$
    $$D_c = \frac{-2}{n} \sum_{i=1}^{n} (y_i - y_{pred})$$
* **Parameter Update / 參數更新**: Updates are applied iteratively using a defined learning rate ($\alpha$). / 根據設定的學習率 $\alpha$ 進行迭代更新。

## 3. Implementation Details / 程式說明
The Python script performs the following steps: / 程式執行步驟如下：
1.  **Data Loading**: Reads `data.csv` using the Pandas library. / 使用 Pandas 讀取資料集。
2.  **Training**: Runs the gradient descent loop for 1,000 iterations to minimize the cost. / 執行 1,000 次梯度下降迭代以最小化誤差。
3.  **Evaluation**: Tracks the cost history to ensure convergence. / 記錄損失值歷史紀錄以確保模型收斂。
4.  **Visualization**: Plots the final regression line and the cost reduction curve. / 繪製最終的迴歸擬合線與損失函數收斂圖。



## 4. How to Run / 如何執行
1.  Ensure you have `numpy`, `pandas`, and `matplotlib` installed. / 確保已安裝必要套件。
2.  Place your dataset in the same directory and name it `data.csv`. / 將資料集命名為 `data.csv` 並放於同目錄。
3.  Run the script / 執行腳本:
    ```bash
    python linear_regression_lab1.py
    ```

---
**Final Results / 最終結果**: 
The script will output the optimized slope ($m$), intercept ($c$), and the final Mean Squared Error.
程式將輸出優化後的斜率、截距以及最終的均方誤差值。