#范權榮 111210557
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 資料集設定
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型管道 (Pipeline)
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# 超參數調校
param_grid = {
    'poly__degree': [1, 2, 3, 4],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# 使用交叉驗證尋找最佳參數
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# 模型評估
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"最佳參數: {grid.best_params_}")
print(f"測試集 RMSE: {rmse:.4f}")
print(f"測試集 R2 分數: {r2:.4f}")

# 驗證曲線視覺化
results_df = pd.DataFrame(grid.cv_results_)
plt.figure(figsize=(10, 6))
for degree in [1, 2, 3]:
    subset = results_df[results_df['param_poly__degree'] == degree]
    plt.plot(subset['param_ridge__alpha'], -subset['mean_test_score'], label=f'Degree {degree}')

plt.xscale('log')
plt.xlabel('Alpha (正則化強度)')
plt.ylabel('均方誤差 (MSE)')
plt.title('模型效能與正則化強度關係圖')
plt.legend()
plt.grid(True)
plt.show()

#范權榮 111210557