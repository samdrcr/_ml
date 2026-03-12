# Machine Learning Assignment: Regularization & Polynomial Regression
# 機器學習作業：正則化與多項式迴歸

**Student Name / 姓名:** 范權榮  
**Student ID / 學號:** 111210557  
**Reference / 參考事項:** GitHub Issue #4 - Addressing Overfitting

---

## Task Overview | 任務概述
The goal of this task was to resolve **overfitting** in a regression model by implementing regularization and feature engineering. The model was optimized to capture non-linear patterns while maintaining high generalization power.

本任務目標是透過實作正則化與特徵工程來解決迴歸模型的**過擬合 (Overfitting)** 問題。模型經過優化後，能捕捉非線性數據趨勢並維持高度的泛化能力。

### Objectives | 核心目標:
* **Feature Expansion (特徵擴展)**: Use `PolynomialFeatures` to capture non-linear trends.
* **Overfitting Control (過擬合控制)**: Implement **Ridge Regression (L2)** to penalize large coefficients.
* **Automated Optimization (自動化優化)**: Use `GridSearchCV` to find the best hyperparameters.
* **Data Validation (數據驗證)**: Apply **5-fold Cross-Validation** to ensure reliability.

---

## Implementation Strategy | 實作策略

I utilized a **Scikit-Learn Pipeline** to ensure a professional workflow and prevent data leakage during scaling and cross-validation.

我使用了 **Scikit-Learn Pipeline (流水線)** 來確保專業的開發流程，並防止在縮放與交叉驗證過程中發生數據洩漏。



### 1. The Pipeline Architecture | 管道架構
1.  **PolynomialFeatures**: Generates high-degree terms (Degrees 1–4 tested).
2.  **StandardScaler**: Normalizes features for uniform Ridge penalty application.
3.  **Ridge Regression**: Minimizes the cost function $RSS + \alpha \sum \beta_j^2$ to reduce variance.

### 2. Hyperparameter Grid | 超參數網格
* **Degrees (階數)**: `[1, 2, 3, 4]`
* **Alpha ($\alpha$ 正則化強度)**: `[0.001, 0.01, 0.1, 1, 10, 100]`

---

## Results & Analysis | 結果與分析

The optimization process yielded a model that balances complexity and stability. The **Validation Curve** illustrates how the Mean Squared Error (MSE) changes with regularization strength.

優化過程產出了一個平衡複雜度與穩定性的模型。透過**驗證曲線 (Validation Curve)**，我們可以看到均方誤差 (MSE) 如何隨著正則化強度的增加而變化。



### Final Performance Metrics | 最終效能指標:
| Metric (指標) | Description (說明) |
| :--- | :--- |
| **Best Degree** | Optimal polynomial degree found via GridSearchCV |
| **Best Alpha ($\alpha$)** | Best regularization strength to prevent overfitting |
| **Test RMSE** | Root Mean Squared Error on unseen data |
| **Test $R^2$ Score** | Proportion of variance explained by the model |

---

## 🚀 Usage | 使用說明

### Prerequisites | 前提條件
```bash
pip install numpy pandas matplotlib scikit-learn
