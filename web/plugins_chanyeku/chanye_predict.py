#import numpy as np
#from sklearn.linear_model import LinearRegression
#
## 假设的 GDP 数据
#years = np.array([2019, 2020, 2021,2022,2023])
#gdp = np.array([90.03, 101.6,111.99,121.00,131.00])  # 单位：万亿
#
## 转换为二维数组
#X = years.reshape(-1, 1)
#
## 使用线性回归模型进行拟合
#model = LinearRegression()
#model.fit(X, gdp)
#
## 预测未来一年的 GDP
#next_year = 2024
#predicted_gdp = model.predict([[next_year]])
#
#print(f"预测 {next_year} 年的 GDP 为 {predicted_gdp[0]:.2f} 万亿")
#



#南阳市社旗县上上年的国内生产总值(gdp)62.88万,上年的国内生产总值(gdp)94.24万,这一年的国内生产总值(gdp)141.23万,来年国内生产总值(gdp)能预测一下吗
#import numpy as np
#from sklearn.linear_model import LinearRegression
#
## 假设的 gdp 数据
#gdp_data = {
#    "前年":62.88,
#    "去年":94.24,
#    "今年":141.23
#}
##years = list(gdp_data.keys())
#years = np.array(list(range(1,len(gdp_data) + 1)))
#gdp = np.array(list(gdp_data.values()))
#
## 转换为二维数组
#X = years.reshape(-1, 1)
#
## 使用线性回归模型进行拟合
#model = LinearRegression()
#model.fit(X, gdp)
#
## 预测未来一年的 gdp
#next_gdp = len(years) + 1
#predicted_gdp = model.predict([[next_gdp]])
#
#print(f"预测明年的gdp为{predicted_gdp[0]:.2f}万")



#临沂沂南上年的国内生产总值(gdp)是11.31千万,这一年的国内生产总值(gdp)是29.63千万,前年的国内生产总值(gdp)是4.32千万,下一年国内生产总值(gdp)预测
import numpy as np
from sklearn.linear_model import LinearRegression

# 假设的 gdp 数据
gdp_data = {
    "前年":4.32,
    "去年":11.31,
    "今年":29.63
}
#years = list(gdp_data.keys())
years = np.array(list(range(1,len(gdp_data) + 1)))
gdp = np.array(list(gdp_data.values()))

# 转换为二维数组
X = years.reshape(-1, 1)

# 使用线性回归模型进行拟合
model = LinearRegression()
model.fit(X, gdp)

# 预测未来一年的 gdp
next_gdp = len(years) + 1
predicted_gdp = model.predict([[next_gdp]])

#import ipdb
#ipdb.set_trace()
print(f"预测明年的gdp为{predicted_gdp[0]:.2f}千万")
