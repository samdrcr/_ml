#范權榮 111210557

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 載入資料集
data = pd.read_csv('data.csv')
X = data.iloc[:, 0].values  # 自動抓取第一欄
y = data.iloc[:, 1].values

# 超參數設定
learning_rate = 0.0001
iterations = 1000

# 初始化參數
m = 0.0
c = 0.0
n = float(len(X))
cost_history = []

# 梯度下降演算法 (Gradient Descent)
for i in range(iterations):
    y_pred = m * X + c
    
    # 計算損失函數 (MSE)
    cost = (1/n) * sum((y - y_pred)**2)
    cost_history.append(cost)
    
    # 計算偏微分 (梯度)
    D_m = (-2/n) * sum(X * (y - y_pred))
    D_c = (-2/n) * sum(y - y_pred)
    
    # 更新權重與偏差
    m = m - learning_rate * D_m
    c = c - learning_rate * D_c

print(f"斜率 m: {m}, 截距 c: {c}, 最終損失值: {cost_history[-1]}")

# 數據視覺化
plt.figure(figsize=(10, 4))

# 迴歸分析圖
plt.subplot(1, 2, 1)
plt.scatter(X, y, s=10, label='數據點')
plt.plot(X, m*X + c, color='red', label='預測線')
plt.title('Linear Regression Fit')
plt.legend()



# 損失函數收斂圖
plt.subplot(1, 2, 2)
plt.plot(cost_history)
plt.title('Cost Convergence')
plt.xlabel('Iterations')
plt.ylabel('Cost')

plt.tight_layout()
plt.show()

#范權榮 111210557