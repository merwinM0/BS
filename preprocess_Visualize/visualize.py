import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import os

# 设置中文字体标签
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 加载处理后的数据
data = pd.read_csv("output/processed_data.csv")

# 确保imgoutput目录存在
os.makedirs("imgoutput", exist_ok=True)

# 1. 数据初步分析：区分分类变量和连续变量
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
continuous_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print("分类变量:", categorical_cols)
print("连续变量:", continuous_cols)

# 关键分类变量的可视化
key_cat_cols = ["WL"]
for col in key_cat_cols:
    plt.figure(figsize=(10, 6))
    data[col].value_counts().plot(kind="bar")
    plt.title(f"{col} 分布")
    plt.xlabel(col)
    plt.ylabel("计数")
    plt.savefig(f"imgoutput/{col}_distribution.png")
    plt.close()

# 关键连续变量的可视化
key_cont_cols = ["PTS", "AST"]
for col in key_cont_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f"{col} 直方图")
    plt.xlabel(col)
    plt.ylabel("频率")
    plt.savefig(f"imgoutput/{col}_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data[col])
    plt.title(f"{col} 箱线图")
    plt.ylabel(col)
    plt.savefig(f"imgoutput/{col}_boxplot.png")
    plt.close()


# 2. 变量相关性分析（可视化）
plt.figure(figsize=(12, 10))
corr_matrix = data[continuous_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("变量相关性热力图")
plt.savefig("imgoutput/correlation_heatmap.png")
plt.close()

# 3. 特征选择
# 假设目标是PTS（或其他，这里以PTS为例）
if "PTS" in continuous_cols:
    X = data[continuous_cols].drop("PTS", axis=1)
    y = data["PTS"]
    selector = SelectKBest(score_func=f_regression, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print("选择的特征:", selected_features)

    # 绘制特征得分
    scores = selector.scores_
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, scores)
    plt.title("特征选择得分")
    plt.xlabel("得分")
    plt.ylabel("特征")
    plt.savefig("imgoutput/feature_selection_scores.png")
    plt.close()

# 4. 特征提取（PCA）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[continuous_cols])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA 特征提取")
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.savefig("imgoutput/pca_features.png")
plt.close()

# 5. 特征编码
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col + "_encoded"] = le.fit_transform(data[col])
    label_encoders[col] = le

# 保存编码后的数据
data.to_csv("output/encoded_data.csv", index=False)

print("可视化完成，图表保存到imgoutput/，编码数据保存到output/encoded_data.csv")
