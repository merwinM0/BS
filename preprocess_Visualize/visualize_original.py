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

def create_original_data_dashboard():
    """创建原始数据分析仪表板"""
    
    # 加载原始数据
    print("正在加载原始数据...")
    try:
        data = pd.read_csv("regular_season_totals_2010_2024.csv")
        print(f"✅ 成功加载原始数据: {data.shape}")
    except Exception as e:
        print(f"❌ 原始数据加载失败: {e}")
        return
    
    # 确保输出目录存在
    os.makedirs("imgoutput", exist_ok=True)
    
    def diagnose_data():
        """诊断原始数据"""
        print("\n=== 原始数据诊断 ===")
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
        print(f"主要数值列: {[col for col in numeric_cols if 'PTS' in col or 'AST' in col or 'REB' in col][:10]}")
        
        return numeric_cols
    
    def create_dashboard(data):
        """创建原始数据仪表板"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '原始得分分布', '得分箱线图（含异常值）',
                '原始特征相关性', '特征重要性分析',
                'PCA降维分析', '比赛结果统计'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter3d"}, {"type": "pie"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # 1. 原始得分分布
        if "PTS" in data.columns:
            pts_data = data["PTS"].dropna()
            fig.add_trace(
                go.Histogram(
                    x=pts_data,
                    name="原始得分分布",
                    nbinsx=50,
                    marker_color='lightblue',
                    hovertemplate='得分: %{x}<br>频次: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            print(f"✅ 原始得分分布: {len(pts_data)} 个数据点")
        
        # 2. 得分箱线图（包含异常值）
        if "PTS" in data.columns:
            fig.add_trace(
                go.Box(
                    y=data["PTS"],
                    name="得分分布（含异常值）",
                    marker_color='lightcoral',
                    hovertemplate='得分: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. 原始特征相关性
        if len(numeric_cols) > 1:
            # 选择主要特征
            main_features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FGM', 'FGA']
            available_features = [col for col in main_features if col in numeric_cols]
            
            if len(available_features) > 1:
                corr_matrix = data[available_features].corr()
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
                print(f"✅ 特征相关性: {len(available_features)} 个主要特征")
        
        # 4. 特征重要性
        if "PTS" in data.columns and len(numeric_cols) > 1:
            X_cols = [col for col in numeric_cols if col != "PTS"]
            X = data[X_cols].fillna(0)
            y = data["PTS"].fillna(data["PTS"].mean())
            
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
                        marker_color='gold',
                        hovertemplate='特征: %{x}<br>F得分: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=2
                )
                print(f"✅ 特征重要性: {len(features)} 个特征")
            except Exception as e:
                print(f"⚠️ 特征重要性分析失败: {e}")
        
        # 5. PCA分析
        if len(numeric_cols) >= 3:
            # 选择主要数值特征
            main_numeric_cols = [col for col in numeric_cols if col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 'FG3A']]
            if len(main_numeric_cols) >= 3:
                X = data[main_numeric_cols].fillna(0)
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
                            size=4,
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
                print(f"✅ PCA分析: {len(main_numeric_cols)} 个主要特征")
        
        # 6. 比赛结果统计
        if "WL" in data.columns:
            wl_counts = data["WL"].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=wl_counts.index,
                    values=wl_counts.values,
                    name="胜负统计",
                    marker_colors=['#FF6B6B', '#4ECDC4'],
                    hovertemplate='结果: %{label}<br>场次: %{value}<br>占比: %{percent}<extra></extra>'
                ),
                row=3, col=2
            )
            print(f"✅ 胜负统计: {dict(wl_counts)}")
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text="NBA原始数据分析仪表板",
                x=0.5,
                font=dict(size=24, color="darkgreen")
            ),
            height=1200,
            showlegend=False,
            template="plotly_white"
        )
        
        # 更新子图标题
        if "PTS" in data.columns:
            fig.update_xaxes(title_text="得分", row=1, col=1)
            fig.update_yaxes(title_text="频次", row=1, col=1)
            fig.update_yaxes(title_text="得分", row=1, col=2)
        
        if len(numeric_cols) > 1:
            fig.update_xaxes(title_text="特征", row=2, col=2, tickangle=45)
            fig.update_yaxes(title_text="F得分", row=2, col=2)
        
        return fig
    
    def create_additional_charts(data):
        """创建额外的原始数据分析图表"""
        charts = {}
        
        # 1. 时间序列分析
        if "GAME_DATE" in data.columns and "PTS" in data.columns:
            try:
                data_copy = data.copy()
                data_copy['GAME_DATE'] = pd.to_datetime(data_copy['GAME_DATE'], errors='coerce')
                valid_data = data_copy.dropna(subset=['GAME_DATE', 'PTS'])
                
                if len(valid_data) > 0:
                    # 按年份统计
                    yearly_stats = valid_data.groupby(valid_data['GAME_DATE'].dt.year)['PTS'].agg(['mean', 'count']).reset_index()
                    
                    charts['time_series'] = go.Figure()
                    charts['time_series'].add_trace(go.Scatter(
                        x=yearly_stats['GAME_DATE'],
                        y=yearly_stats['mean'],
                        mode='lines+markers',
                        name='年平均得分',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    charts['time_series'].add_trace(go.Scatter(
                        x=yearly_stats['GAME_DATE'],
                        y=yearly_stats['count'],
                        mode='lines+markers',
                        name='年比赛场次',
                        yaxis='y2',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    charts['time_series'].update_layout(
                        title='历年得分趋势与比赛场次',
                        xaxis_title='年份',
                        yaxis_title='平均得分',
                        yaxis2=dict(title='比赛场次', overlaying='y', side='right'),
                        template='plotly_white'
                    )
                    print(f"✅ 时间序列分析: {len(yearly_stats)} 年数据")
            except Exception as e:
                print(f"⚠️ 时间序列分析失败: {e}")
        
        # 2. 球队表现对比
        if "TEAM_ABBREVIATION" in data.columns and "PTS" in data.columns:
            try:
                team_stats = data.groupby('TEAM_ABBREVIATION')['PTS'].agg(['mean', 'count', 'std']).reset_index()
                team_stats = team_stats[team_stats['count'] > 50].sort_values('mean', ascending=False)
                
                if len(team_stats) > 0:
                    charts['team_comparison'] = go.Figure()
                    charts['team_comparison'].add_trace(go.Bar(
                        x=team_stats['TEAM_ABBREVIATION'],
                        y=team_stats['mean'],
                        name='平均得分',
                        marker_color='lightcoral',
                        error_y=dict(type='data', array=team_stats['std'])
                    ))
                    charts['team_comparison'].update_layout(
                        title='各球队平均得分对比（含标准差）',
                        xaxis_title='球队',
                        yaxis_title='平均得分',
                        template='plotly_white'
                    )
                    print(f"✅ 球队对比: {len(team_stats)} 支球队")
            except Exception as e:
                print(f"⚠️ 球队对比分析失败: {e}")
        
        return charts
    
    # 诊断数据
    numeric_cols = diagnose_data()
    
    if data.empty:
        print("❌ 数据为空，无法生成报告")
        return
    
    # 创建主仪表板
    print("\n正在生成原始数据分析图表...")
    main_dashboard = create_dashboard(data)
    
    # 创建额外图表
    additional_charts = create_additional_charts(data)
    
    # 生成HTML报告
    print("正在生成HTML报告...")
    
    try:
        main_json = main_dashboard.to_json()
    except Exception as e:
        print(f"主仪表板JSON生成失败: {e}")
        main_json = '{"data": [], "layout": {"title": "图表生成失败"}}'
    
    chart_jsons = {}
    for chart_name, chart in additional_charts.items():
        try:
            chart_jsons[chart_name] = chart.to_json()
        except Exception as e:
            print(f"{chart_name} JSON生成失败: {e}")
            chart_jsons[chart_name] = '{"data": [], "layout": {"title": "图表生成失败"}}'
    
    # 生成HTML内容
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NBA原始数据分析报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-container {{ margin-bottom: 30px; background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-stats {{ display: flex; justify-content: space-around; margin-bottom: 30px; flex-wrap: wrap; }}
        .stat-box {{ background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 150px; margin: 10px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .data-info {{ background-color: #e8f4fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NBA原始数据分析报告</h1>
        <p>基于2010-2024赛季原始数据的深度分析（未经过预处理）</p>
    </div>
    
    <div class="data-info">
        <h3>📊 数据概览</h3>
        <p><strong>数据源:</strong> regular_season_totals_2010_2024.csv</p>
        <p><strong>数据特点:</strong> 包含原始异常值、缺失值，未经标准化处理</p>
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
                <div class="stat-label">总记录数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_pts:.1f}</div>
                <div class="stat-label">平均得分</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{num_features}</div>
                <div class="stat-label">原始特征数</div>
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
        <h2>🔍 原始数据综合分析仪表板</h2>
        <div id="main-dashboard"></div>
    </div>"""
    
    # 添加额外图表
    chart_titles = {
        'time_series': '📈 历年得分趋势分析',
        'team_comparison': '🏀 球队表现对比分析'
    }
    
    for chart_name in additional_charts.keys():
        title = chart_titles.get(chart_name, chart_name)
        html_content += f"""
    <div class="chart-container">
        <h2>{title}</h2>
        <div id="{chart_name}"></div>
    </div>"""
    
    # 添加JavaScript
    html_content += f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('开始渲染原始数据图表...');
            
            try {{
                var mainData = {main_json};
                Plotly.newPlot('main-dashboard', mainData.data, mainData.layout, {{responsive: true}});
                console.log('主仪表板渲染成功');
            }} catch(e) {{
                console.error('主仪表板渲染失败:', e);
            }}"""
    
    for chart_name in additional_charts.keys():
        html_content += f"""
            
            try {{
                var {chart_name}Data = {chart_jsons[chart_name]};
                Plotly.newPlot('{chart_name}', {chart_name}Data.data, {chart_name}Data.layout, {{responsive: true}});
                console.log('{chart_name}图表渲染成功');
            }} catch(e) {{
                console.error('{chart_name}图表渲染失败:', e);
            }}"""
    
    html_content += """
        });
    </script>
</body>
</html>"""
    
    # 保存文件
    output_file = "imgoutput/nba_original_data_analysis.html"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 同时保存主仪表板
        main_dashboard.write_html("imgoutput/original_main_dashboard.html", include_plotlyjs='cdn')
        
        print(f"\n✅ 原始数据分析报告已生成: {output_file}")
        print("📊 报告特点:")
        print("   - 包含原始异常值分析")
        print("   - 未经过数据标准化的真实分布")
        print("   - 原始特征相关性分析")
        print("   - 完整的数据质量评估")
        print(f"\n🎯 查看文件:")
        print(f"   - 完整报告: {output_file}")
        print("   - 主仪表板: imgoutput/original_main_dashboard.html")
        
    except Exception as e:
        print(f"❌ 文件保存失败: {e}")

if __name__ == "__main__":
    create_original_data_dashboard()
