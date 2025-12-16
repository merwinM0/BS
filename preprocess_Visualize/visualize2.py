import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import json

# 设置中文字体
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 加载原始数据（不使用预处理后的数据）
print("正在加载数据...")
try:
    # 先尝试加载原始数据
    data = pd.read_csv("regular_season_totals_2010_2024.csv")
    print(f"✅ 成功加载原始数据: {data.shape}")
except:
    # 如果原始数据不存在，使用预处理后的数据
    data = pd.read_csv("output/processed_data.csv")
    print(f"⚠️ 使用预处理数据: {data.shape}")

# 确保输出目录存在
os.makedirs("imgoutput", exist_ok=True)

def diagnose_data_issues():
    """诊断数据问题"""
    print("\n=== 数据诊断 ===")
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    
    # 检查关键列
    key_columns = ['PTS', 'WL', 'GAME_DATE', 'TEAM_ABBREVIATION']
    for col in key_columns:
        if col in data.columns:
            print(f"✅ {col}: 存在, 非空值: {data[col].notna().sum()}/{len(data)}")
            if data[col].dtype in ['int64', 'float64']:
                print(f"   范围: {data[col].min():.2f} - {data[col].max():.2f}")
        else:
            print(f"❌ {col}: 不存在")
    
    # 检查数值列
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n数值列数量: {len(numeric_cols)}")
    print(f"数值列: {numeric_cols[:10]}...")  # 只显示前10个
    
    return numeric_cols

def create_comprehensive_dashboard(data):
    """创建综合动态仪表板"""
    
    # 获取数值列
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建子图布局
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '得分分布直方图', '得分箱线图',
            '变量相关性热力图', '特征选择得分',
            'PCA降维可视化', '比赛结果分布'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "box"}],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "scatter3d"}, {"type": "pie"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )
    
    # 1. 得分分布直方图
    if "PTS" in data.columns and not data["PTS"].isna().all():
        pts_data = data["PTS"].dropna()
        if len(pts_data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=pts_data,
                    name="得分分布",
                    nbinsx=30,
                    marker_color='skyblue',
                    hovertemplate='得分: %{x}<br>频次: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            print(f"✅ 得分直方图: {len(pts_data)} 个数据点")
        else:
            print("⚠️ PTS列全为空，跳过得分直方图")
    
    # 2. 得分箱线图
    if "PTS" in data.columns and not data["PTS"].isna().all():
        pts_data = data["PTS"].dropna()
        if len(pts_data) > 0:
            fig.add_trace(
                go.Box(
                    y=pts_data,
                    name="得分箱线图",
                    marker_color='lightgreen',
                    hovertemplate='得分: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
            print(f"✅ 得分箱线图: {len(pts_data)} 个数据点")
    
    # 3. 相关性热力图
    if len(numeric_cols) > 1:
        # 过滤掉常数列
        valid_numeric_cols = []
        for col in numeric_cols:
            if data[col].nunique() > 1:  # 不是常数列
                valid_numeric_cols.append(col)
        
        if len(valid_numeric_cols) > 1:
            corr_matrix = data[valid_numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name="相关性",
                    hovertemplate='变量1: %{x}<br>变量2: %{y}<br>相关系数: %{z:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            print(f"✅ 相关性热力图: {len(valid_numeric_cols)} 个变量")
        else:
            print("⚠️ 有效数值列不足，跳过相关性热力图")
    
    # 4. 特征选择得分
    if "PTS" in data.columns and len(numeric_cols) > 1:
        pts_data = data["PTS"].dropna()
        if len(pts_data) > 0:
            X_cols = [col for col in numeric_cols if col != "PTS"]
            X = data[X_cols].fillna(0)  # 填充缺失值
            y = pts_data
            
            try:
                selector = SelectKBest(score_func=f_regression, k=min(10, len(X_cols)))
                selector.fit(X, y)
                scores = selector.scores_
                features = X.columns
                
                fig.add_trace(
                    go.Bar(
                        x=features,
                        y=scores,
                        name="特征重要性",
                        marker_color='orange',
                        hovertemplate='特征: %{x}<br>F得分: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=2
                )
                print(f"✅ 特征选择: {len(features)} 个特征")
            except Exception as e:
                print(f"⚠️ 特征选择失败: {e}")
    
    # 5. PCA 3D可视化
    if len(numeric_cols) >= 3:
        # 过滤有效列
        valid_cols = []
        for col in numeric_cols[:10]:  # 最多取前10个数值列
            if data[col].nunique() > 1 and not data[col].isna().all():
                valid_cols.append(col)
        
        if len(valid_cols) >= 3:
            try:
                X = data[valid_cols].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X_scaled)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        z=X_pca[:, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=X_pca[:, 2],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="PC3值", x=0.45)
                        ),
                        name="PCA",
                        text=[f'PC1:{x:.2f}<br>PC2:{y:.2f}<br>PC3:{z:.2f}' 
                              for x, y, z in zip(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=3, col=1
                )
                print(f"✅ PCA可视化: {len(valid_cols)} 个变量")
            except Exception as e:
                print(f"⚠️ PCA失败: {e}")
        else:
            print("⚠️ 有效数值列不足3个，跳过PCA")
    
    # 6. 比赛结果分布饼图
    if "WL" in data.columns and not data["WL"].isna().all():
        wl_counts = data["WL"].value_counts()
        if len(wl_counts) > 0:
            fig.add_trace(
                go.Pie(
                    labels=wl_counts.index,
                    values=wl_counts.values,
                    name="胜负分布",
                    marker_colors=['#FF9999', '#66B2FF'],
                    hovertemplate='结果: %{label}<br>场次: %{value}<br>占比: %{percent}<extra></extra>'
                ),
                row=3, col=2
            )
            print(f"✅ 胜负分布: {dict(wl_counts)}")
    
    # 更新整体布局
    fig.update_layout(
        title=dict(
            text="NBA数据分析综合仪表板",
            x=0.5,
            font=dict(size=24, color="darkblue")
        ),
        height=1200,
        showlegend=False,
        template="plotly_white"
    )
    
    # 更新子图布局
    if "PTS" in data.columns:
        fig.update_xaxes(title_text="得分", row=1, col=1)
        fig.update_yaxes(title_text="频次", row=1, col=1)
        fig.update_yaxes(title_text="得分", row=1, col=2)
    
    if "PTS" in data.columns and len(numeric_cols) > 1:
        fig.update_xaxes(title_text="特征", row=2, col=2, tickangle=45)
        fig.update_yaxes(title_text="F得分", row=2, col=2)
    
    if len(numeric_cols) >= 3:
        fig.update_xaxes(title_text="PC1", row=3, col=1)
        fig.update_yaxes(title_text="PC2", row=3, col=1)
    
    return fig

# 主执行函数
def main():
    print("=== NBA数据分析动态报告生成器 ===")
    
    # 诊断数据
    numeric_cols = diagnose_data_issues()
    
    # 检查数据
    if data.empty:
        print("❌ 数据为空，无法生成报告")
        return
    
    # 生成并保存主仪表板
    main_dashboard = create_comprehensive_dashboard(data)
    main_dashboard.write_html("imgoutput/main_dashboard.html", include_plotlyjs='cdn')
    
    print(f"\n✅ 主仪表板已生成: imgoutput/main_dashboard.html")
    print("📊 仪表板包含以下内容:")
    print("   - 得分分布直方图")
    print("   - 得分箱线图")
    print("   - 变量相关性热力图")
    print("   - 特征选择得分")
    print("   - PCA降维可视化")
    print("   - 比赛结果分布")
    
    print("\n🎯 可以直接在浏览器中打开查看:")
    print("   - 主仪表板: imgoutput/main_dashboard.html")

if __name__ == "__main__":
    main()
