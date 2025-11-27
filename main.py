import pandas as pd

# 读取CSV文件
df = pd.read_csv('../NBA-Data-2010-2024-main/play_off_totals_2010_2024.csv')

# 显示数据行数
print(f"行数: {len(df)}")
