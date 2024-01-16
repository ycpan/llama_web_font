import numpy as np
from sklearn.linear_model import LinearRegression

# 假设的 GDP 数据
years = np.array([2019, 2020, 2021,2022,2023])
gdp = np.array([90.03, 101.6,111.99,121.00,131.00])  # 单位：万亿

# 转换为二维数组
X = years.reshape(-1, 1)

# 使用线性回归模型进行拟合
model = LinearRegression()
model.fit(X, gdp)

# 预测未来一年的 GDP
next_year = 2024
predicted_gdp = model.predict([[next_year]])

print(f"预测 {next_year} 年的 GDP 为 {predicted_gdp[0]:.2f} 万亿")

