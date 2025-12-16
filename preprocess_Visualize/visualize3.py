import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
import os

# 设置中文字体
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 加载相关性矩阵数据
print("正在加载相关性矩阵数据...")
corr_matrix = pd.read_csv("./output/correlation_matrix.csv", index_col=0)

# 移除空行和空列
corr_matrix = corr_matrix.dropna(how='all').dropna(axis=1, how='all')

# 确保输出目录存在
os.makedirs("imgoutput", exist_ok=True)

def create_correlation_network():
    """创建相关性网络图"""
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for col in corr_matrix.columns:
        G.add_node(col)
    
    # 添加边（只显示强相关性）
    threshold = 0.7
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                G.add_edge(corr_matrix.columns[i], 
                          corr_matrix.columns[j], 
                          weight=abs(corr_value))
    
    # 使用spring布局
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # 创建plotly网络图
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_info.append(f"变量: {node}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        hovertext=node_info,
        textposition="middle center",
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='强相关性网络图 (|r| > 0.7)',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def create_clustered_heatmap():
    """创建聚类热力图"""
    # 进行层次聚类
    linkage_matrix = linkage(corr_matrix, method='average')
    
    # 创建聚类热力图
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        hovertemplate='变量1: %{x}<br>变量2: %{y}<br>相关系数: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='聚类相关性热力图',
            font=dict(size=16)
        ),
        xaxis_title='变量',
        yaxis_title='变量',
        width=800,
        height=800
    )
    
    return fig

def create_mds_visualization():
    """创建多维标度可视化"""
    # 将相关性转换为距离
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # 使用MDS降维
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    
    # 修复：将range转换为list
    colors = list(range(len(corr_matrix.columns)))
    
    # 创建MDS散点图
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        text=corr_matrix.columns,
        textposition="top center",
        marker=dict(
            size=12,
            color=colors,  # 修复：使用列表而不是range对象
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="变量索引")
        ),
        hovertemplate='变量: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='多维标度(MDS)可视化 - 变量关系映射',
            font=dict(size=16)
        ),
        xaxis_title='维度1',
        yaxis_title='维度2',
        width=800,
        height=600
    )
    
    return fig

def create_correlation_distribution():
    """创建相关性分布图"""
    # 提取上三角矩阵的相关系数（排除对角线）
    corr_values = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_values.append(corr_matrix.iloc[i, j])
    
    # 创建分布图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('相关性系数直方图', '相关性系数箱线图', 
                       '正相关性分布', '负相关性分布'),
        specs=[[{"type": "histogram"}, {"type": "box"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # 总体分布直方图
    fig.add_trace(
        go.Histogram(x=corr_values, name="总体分布", nbinsx=30, 
                    marker_color='skyblue'),
        row=1, col=1
    )
    
    # 箱线图
    fig.add_trace(
        go.Box(y=corr_values, name="箱线图", marker_color='lightgreen'),
        row=1, col=2
    )
    
    # 正相关性分布
    pos_corr = [x for x in corr_values if x > 0]
    fig.add_trace(
        go.Histogram(x=pos_corr, name="正相关", nbinsx=20, 
                    marker_color='orange'),
        row=2, col=1
    )
    
    # 负相关性分布
    neg_corr = [x for x in corr_values if x < 0]
    fig.add_trace(
        go.Histogram(x=neg_corr, name="负相关", nbinsx=20, 
                    marker_color='red'),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(
            text='相关性系数分布分析',
            font=dict(size=16)
        ),
        height=600,
        showlegend=False
    )
    
    return fig

def create_top_correlations_bar():
    """创建Top相关性条形图"""
    # 获取所有相关性对
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlations.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'corr': corr_matrix.iloc[i, j]
            })
    
    # 按绝对值排序
    correlations.sort(key=lambda x: abs(x['corr']), reverse=True)
    
    # 取前15个
    top_corr = correlations[:15]
    
    # 创建标签
    labels = [f"{row['var1']} - {row['var2']}" for row in top_corr]
    values = [row['corr'] for row in top_corr]
    colors = ['red' if x < 0 else 'green' for x in values]
    
    # 创建条形图
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=colors,
            hovertemplate='变量对: %{y}<br>相关系数: %{x:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Top 15 最强相关性',
            font=dict(size=16)
        ),
        xaxis_title='相关系数',
        yaxis_title='变量对',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def main():
    print("=== 相关性矩阵可视化分析 ===")
    print(f"数据维度: {corr_matrix.shape}")
    
    # 1. 相关性网络图
    network_fig = create_correlation_network()
    network_fig.write_html("imgoutput/correlation_network.html")
    print("✅ 相关性网络图已生成: imgoutput/correlation_network.html")
    
    # 2. 聚类热力图
    cluster_fig = create_clustered_heatmap()
    cluster_fig.write_html("imgoutput/clustered_heatmap.html")
    print("✅ 聚类热力图已生成: imgoutput/clustered_heatmap.html")
    
    # 3. MDS可视化
    mds_fig = create_mds_visualization()
    mds_fig.write_html("imgoutput/mds_visualization.html")
    print("✅ MDS可视化已生成: imgoutput/mds_visualization.html")
    
    # 4. 相关性分布分析
    dist_fig = create_correlation_distribution()
    dist_fig.write_html("imgoutput/correlation_distribution.html")
    print("✅ 相关性分布分析已生成: imgoutput/correlation_distribution.html")
    
    # 5. Top相关性条形图
    top_fig = create_top_correlations_bar()
    top_fig.write_html("imgoutput/top_correlations.html")
    print("✅ Top相关性条形图已生成: imgoutput/top_correlations.html")
    
    print("\n🎯 所有可视化文件已生成完成！")
    print("📊 生成的图表包括:")
    print("   - 相关性网络图 (显示强相关关系)")
    print("   - 聚类热力图 (层次聚类版本)")
    print("   - MDS可视化 (多维标度映射)")
    print("   - 相关性分布分析 (统计分布)")
    print("   - Top相关性条形图 (最强相关对)")

if __name__ == "__main__":
    main()
