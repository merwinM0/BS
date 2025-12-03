import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import os

# Load the CSV file
data = pd.read_csv("regular_season_totals_2010_2024.csv")

# Data Quality Assessment
print("Initial Data Quality Assessment:")
print(data.info())
print(data.describe())
print("Missing values per column:")
print(data.isnull().sum())

# 1. Missing Value Handling
# Fill numerical columns with mean, categorical with mode
for col in data.columns:
    if data[col].dtype in ["int64", "float64"]:
        data[col].fillna(data[col].mean(), inplace=True)
    else:
        data[col].fillna(
            data[col].mode()[0] if not data[col].mode().empty else "Unknown",
            inplace=True,
        )

# 2. Outlier Handling using IQR for numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# 3. Discretization of continuous data (e.g., PTS column if exists, assuming a points column)
if "PTS" in data.columns:
    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    data["PTS_discretized"] = discretizer.fit_transform(data[["PTS"]])

# 4. Standardization
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# 5. Correlation Analysis
correlation_matrix = data[numerical_cols].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# 6. Final Data Quality Assessment
print("Final Data Quality Assessment:")
print(data.info())
print(data.describe())

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Save processed data
data.to_csv("output/processed_data.csv", index=False)

# Save correlation matrix
correlation_matrix.to_csv("output/correlation_matrix.csv")

print("Preprocessing complete. Processed data saved to output/processed_data.csv")
