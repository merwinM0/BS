import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import os

data = pd.read_csv("regular_season_totals_2010_2024.csv")

print("初始数据质量评估:")
print(data.info())
print(data.describe())
print("每列缺失值:")
print(data.isnull().sum())

for col in data.columns:
    if data[col].dtype in ["int64", "float64"]:
        data[col].fillna(data[col].mean(), inplace=True)
    else:
        data[col].fillna(
            data[col].mode()[0] if not data[col].mode().empty else "未知",
            inplace=True,
        )

numerical_cols = data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

if "PTS" in data.columns:
    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    data["PTS_discretized"] = discretizer.fit_transform(data[["PTS"]])

scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

correlation_matrix = data[numerical_cols].corr()
print("相关性矩阵:")
print(correlation_matrix)

print("最终数据质量评估:")
print(data.info())
print(data.describe())

os.makedirs("output", exist_ok=True)

data.to_csv("output/processed_data.csv", index=False)

correlation_matrix.to_csv("output/correlation_matrix.csv")

print("预处理完成。处理后的数据已保存至 output/processed_data.csv")
