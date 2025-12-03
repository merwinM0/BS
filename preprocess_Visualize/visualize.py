import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import os

# Set font for Chinese labels
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# Load processed data
data = pd.read_csv("output/processed_data.csv")

# Ensure imgoutput directory exists
os.makedirs("imgoutput", exist_ok=True)

# 1. Data preliminary analysis: Distinguish categorical and continuous variables
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
continuous_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print("分类变量:", categorical_cols)
print("连续变量:", continuous_cols)

# Visualization for key categorical variables
key_cat_cols = ["WL"]
for col in key_cat_cols:
    plt.figure(figsize=(10, 6))
    data[col].value_counts().plot(kind="bar")
    plt.title(f"{col} 分布")
    plt.xlabel(col)
    plt.ylabel("计数")
    plt.savefig(f"imgoutput/{col}_distribution.png")
    plt.close()

# Visualization for key continuous variables
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


# 2. Variable correlation analysis (visualization)
plt.figure(figsize=(12, 10))
corr_matrix = data[continuous_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("变量相关性热力图")
plt.savefig("imgoutput/correlation_heatmap.png")
plt.close()

# 3. Feature selection
# Assume target is PTS (or another, here using PTS as example)
if "PTS" in continuous_cols:
    X = data[continuous_cols].drop("PTS", axis=1)
    y = data["PTS"]
    selector = SelectKBest(score_func=f_regression, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print("选择的特征:", selected_features)

    # Plot feature scores
    scores = selector.scores_
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, scores)
    plt.title("特征选择得分")
    plt.xlabel("得分")
    plt.ylabel("特征")
    plt.savefig("imgoutput/feature_selection_scores.png")
    plt.close()

# 4. Feature extraction (PCA)
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

# 5. Feature encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col + "_encoded"] = le.fit_transform(data[col])
    label_encoders[col] = le

# Save encoded data
data.to_csv("output/encoded_data.csv", index=False)

print("可视化完成，图表保存到imgoutput/，编码数据保存到output/encoded_data.csv")
