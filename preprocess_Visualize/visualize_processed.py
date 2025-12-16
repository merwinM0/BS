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

def create_processed_data_dashboard():
    """创建处理后数据分析仪表板"""
    
    # 加载处理后的数据
    print("正在加载处理后数据...")
    try:
        data = pd.read_csv("output/processed_data.csv")
        print(f"✅ 成功加载处理后数据: {data.shape}")
    except Exception as e:
        print(f"❌ 处理后数据加载失败: {e}")
        return
    
    # 确保输出目录存在
    os.makedirs("imgoutput", exist_ok=True)
    
    def diagnose_processed_data():
        """诊断处理后数据"""
        print("\n=== 处理后数据诊断 ===")
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
        print(f"数值列: {numeric_cols[:10]}...")
        
        return numeric_cols
    
    def create_dashboard(data):
        """创建处理后数据仪表板"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '标准化得分分布', '处理后得分箱线图',
                '处理后特征相关性', '特征重要性（标准化数据）',
                'PCA分析（标准化）', '比赛结果分布'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter3d"}, {"type": "pie"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # 1. 标准化得分分布
        if "PTS" in data.columns:
            pts_data = data["PTS"].dropna()
            fig.add_trace(
                go.Histogram(
                    x=pts_data,
                    name="标准化得分分布",
                    nbinsx=30,
                    marker_color='lightgreen',
                    hovertemplate='标准化得分: %{x}<br>频次: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            print(f"✅ 标准化得分分布: {len(pts_data)} 个数据点")
        
        # 2. 处理后得分箱线图
        if "PTS" in data.columns:
            fig.add_trace(
                go.Box(
                    y=data["PTS"],
                    name="处理后得分分布",
                    marker_color='lightblue',
                    hovertemplate='标准化得分: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. 处理后特征相关性
        if len(numeric_cols) > 1:
            # 过滤常数列
            valid_numeric_cols = []
            for col in numeric_cols:
                if data[col].nunique() > 1:
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
                        name="特征相关性",
                        hovertemplate='变量1: %{x}<br>变量2: %{y}<br>相关系数: %{z:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                print(f"✅ 处理后特征相关性: {len(valid_numeric_cols)} 个变量")
        
        # 4. 特征重要性（基于标准化数据）
        if "PTS" in data.columns and len(numeric_cols) > 1:
            X_cols = [col for col in numeric_cols if col != "PTS"]
            X = data[X_cols].fillna(0)
            y = data["PTS"].fillna(0)
            
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
                print(f"✅ 特征重要性分析: {len(features)} 个特征")
            except Exception as e:
                print(f"⚠️ 特征重要性分析失败: {e}")
        
        # 5. PCA分析（基于标准化数据）
        if len(numeric_cols) >= 3:
            valid_cols = []
            for col in numeric_cols[:10]:
                if data[col].nunique() > 1 and not data[col].isna().all():
                    valid_cols.append(col)
            
            if len(valid_cols) >= 3:
                X = data[valid_cols].fillna(0)
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X)
                
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
                        name="PCA分析",
                        text=[f'PC1:{x:.2f}<br>PC2:{y:.2f}<br>PC3:{z:.2f}' 
                              for x, y, z in zip(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=3, col=1
                )
                print(f"✅ PCA分析: {len(valid_cols)} 个标准化特征")
        
        # 6. 比赛结果分布
        if "WL" in data.columns:
            wl_counts = data["WL"].value_counts()
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
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text="NBA处理后数据分析仪表板",
                x=0.5,
                font=dict(size=24, color="darkblue")
            ),
            height=1200,
            showlegend=False,
            template="plotly_white"
        )
        
        # 更新子图标题
        if "PTS" in data.columns:
            fig.update_xaxes(title_text="标准化得分", row=1, col=1)
            fig.update_yaxes(title_text="频次", row=1, col=1)
            fig.update_yaxes(title_text="标准化得分", row=1, col=2)
        
        if len(numeric_cols) > 1:
            fig.update_xaxes(title_text="特征", row=2, col=2, tickangle=45)
            fig.update_yaxes(title_text="F得分", row=2, col=2)
        
        return fig
    
    # 诊断数据
    numeric_cols = diagnose_processed_data()
    
    if data.empty:
        print("❌ 数据为空，无法生成报告")
        return
    
    # 创建主仪表板
    print("\n正在生成处理后数据分析图表...")
    main_dashboard = create_dashboard(data)
    
    # 生成HTML报告
    print("正在生成HTML报告...")
    
    try:
        main_json = main_dashboard.to_json()
    except Exception as e:
        print(f"主仪表板JSON生成失败: {e}")
        main_json = '{"data": [], "layout": {"title": "图表生成失败"}}'
    
    # 生成HTML内容
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NBA处理后数据分析报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-container {{ margin-bottom: 30px; background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-stats {{ display: flex; justify-content: space-around; margin-bottom: 30px; flex-wrap: wrap; }}
        .stat-box {{ background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 150px; margin: 10px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .data-info {{ background-color: #e8f8e8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NBA处理后数据分析报告</h1>
        <p>基于2010-2024赛季处理后数据的分析（经过标准化、异常值处理）</p>
    </div>
    
    <div class="data-info">
        <h3>📊 数据处理说明</h3>
        <p><strong>数据源:</strong> output/processed_data.csv</p>
        <p><strong>处理步骤:</strong> 缺失值填充 → 异常值处理(IQR) → 数据标准化 → 特征离散化</p>
        <p><strong>数据特点:</strong> 标准化处理、异常值已移除、适合机器学习建模</p>
    </div>
    
    <div class="summary-stats">"""
    
    # 添加统计信息
    try:
        total_records = len(data)
        avg_pts = data['PTS'].mean() if 'PTS' in data.columns else 0
        num_features = data.select_dtypes(include=[np.number]).shape[1]
        num_teams = data['TEAM_ABBREVIATION'].nunique() if 'TEAM_ABBREVIATION' in data.columns else 0
        
        html_content += f"""
            <div class="stat-box">
                <div class="stat-value">{total_records:,}</div>
                <div class="stat-label">处理后记录数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_pts:.2f}</div>
                <div class="stat-label">标准化平均得分</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{num_features}</div>
                <div class="stat-label">处理后特征数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{num_teams}</div>
                <div class="stat-label">球队数量</div>
            </div>"""
    except Exception as e:
        print(f"统计信息生成失败: {e}")
    
    html_content += """
    </div>
    
    <div class="chart-container">
        <h2>🔧 处理后数据综合分析仪表板</h2>
        <div id="main-dashboard"></div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('开始渲染处理后数据图表...');
            
            try {{
                var mainData = {main_json};
                Plotly.newPlot('main-dashboard', mainData.data, mainData.layout, {{responsive: true}});
                console.log('处理后数据主仪表板渲染成功');
            }} catch(e) {{
                console.error('主仪表板渲染失败:', e);
            }}
        }});
    </script>
</body>
</html>"""
    
    # 保存文件
    output_file = "imgoutput/nba_processed_data_analysis.html"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 保存main_dashboard.html（保持不变）
        main_dashboard.write_html("imgoutput/main_dashboard.html", include_plotlyjs='cdn')
        
        print(f"\n✅ 处理后数据分析报告已生成: {output_file}")
        print("📊 报告特点:")
        print("   - 标准化后的数据分布")
        print("   - 异常值处理后的清洁数据")
        print("   - 适合机器学习的特征分析")
        print("   - 数据预处理效果评估")
        print(f"\n🎯 查看文件:")
        print(f"   - 完整报告: {output_file}")
        print("   - 主仪表板: imgoutput/main_dashboard.html")
        
    except Exception as e:
        print(f"❌ 文件保存失败: {e}")

if __name__ == "__main__":
    create_processed_data_dashboard()
