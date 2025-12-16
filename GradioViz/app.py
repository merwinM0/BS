import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ==========数据加载函数==========
def load_data():
    try:
        df = pd.read_csv('regular_season_totals_2010_2024.csv')
        return df
    except FileNotFoundError:
        print("请确保文件 'regular_season_totals_2010_2024.csv' 在当前目录下")
        return None

# ==========字体设置函数==========
def set_font():
    import matplotlib
    import matplotlib.font_manager as fm
    
    font_found = False
    for font in fm.fontManager.ttflist:
        if 'Noto Sans CJK JP' in font.name:
            matplotlib.rcParams['font.sans-serif'] = [font.name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            font_found = True
            print(f"使用字体: {font.name}")
            break
    
    if not font_found:
        raise Exception("未找到Noto Sans CJK JP字体，请安装该字体")
    
    return matplotlib.rcParams['font.sans-serif'][0]

# ==========字段显示管理函数==========
def get_display_fields(df, selected_field, field_type, scroll_offset=0):
    all_fields = df.columns.tolist()
    
    if selected_field not in all_fields:
        selected_field = all_fields[0]
    
    idx = all_fields.index(selected_field)
    adjusted_idx = idx + scroll_offset
    
    if adjusted_idx < 0:
        adjusted_idx = 0
    elif adjusted_idx >= len(all_fields):
        adjusted_idx = len(all_fields) - 1
    
    start_idx = max(0, adjusted_idx - 2)
    end_idx = min(len(all_fields), start_idx + 5)
    
    if end_idx - start_idx < 5:
        start_idx = max(0, end_idx - 5)
    
    display_fields = all_fields[start_idx:end_idx]
    
    if selected_field not in display_fields and selected_field in all_fields:
        idx = all_fields.index(selected_field)
        start_idx = max(0, idx - 2)
        end_idx = min(len(all_fields), start_idx + 5)
        if end_idx - start_idx < 5:
            start_idx = max(0, end_idx - 5)
        display_fields = all_fields[start_idx:end_idx]
    
    return display_fields, selected_field, start_idx


















# ==========图表绘制核心函数==========
def plot_four_basic_charts(df, category_field, analysis_field):
    if df is None or len(df) == 0:
        return None
    
    font_name = set_font()
    
    if category_field not in df.columns or analysis_field not in df.columns:
        return None
    
    data = df[[category_field, analysis_field]].dropna().copy()
    if len(data) < 1:
        return None
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ==========直方图绘制==========
    ax1 = fig.add_subplot(gs[0, 0])
    top_categories = data[category_field].value_counts().head(10).index
    filtered_data = data[data[category_field].isin(top_categories)].copy()
    
    if len(filtered_data) > 0:
        categories = filtered_data[category_field].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        data_range = (filtered_data[analysis_field].min(), filtered_data[analysis_field].max())
        bins = 20
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            category_data = filtered_data[filtered_data[category_field] == category][analysis_field]
            if len(category_data) > 0:
                ax1.hist(category_data, bins=bins, alpha=0.5, 
                        label=str(category), color=color, 
                        range=data_range, edgecolor='black')
        
        ax1.set_xlabel(analysis_field, fontsize=11, fontname=font_name)
        ax1.set_ylabel('数量', fontsize=11, fontname=font_name)
        ax1.set_title(f'直方图: 按 {category_field} 分组的 {analysis_field} 数量', 
                     fontsize=12, fontweight='bold', fontname=font_name)
        ax1.legend(title=category_field, fontsize=9, title_fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, '无有效数据', 
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=10, fontname=font_name)
        ax1.set_title('直方图', fontsize=12, fontweight='bold', fontname=font_name)
    
    # ==========饼图绘制==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    if data[analysis_field].dtype.kind in 'iuf':
        grouped = data.groupby(category_field)[analysis_field].mean()
    else:
        grouped = data.groupby(category_field).size()
    
    if len(grouped) > 0:
        top_groups = grouped.nlargest(10)
        wedges, texts, autotexts = ax2.pie(top_groups.values, 
                                          labels=top_groups.index,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=plt.cm.tab20c(np.linspace(0, 1, len(top_groups))))
        
        ax2.set_title(f'饼图: 按 {category_field} 分组的 {analysis_field} 占比', 
                     fontsize=12, fontweight='bold', fontname=font_name)
        
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
    else:
        ax2.text(0.5, 0.5, '无有效数据', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=10, fontname=font_name)
        ax2.set_title('饼图', fontsize=12, fontweight='bold', fontname=font_name)
    
    # ==========核密度估计图绘制==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    if data[analysis_field].dtype.kind in 'iuf' and len(filtered_data) > 0:
        from scipy.stats import gaussian_kde
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(categories), 8)))
        
        for i, (category, color) in enumerate(zip(categories[:8], colors)):
            category_data = filtered_data[filtered_data[category_field] == category][analysis_field]
            if len(category_data) > 1:
                try:
                    kde = gaussian_kde(category_data)
                    x_range = np.linspace(data_range[0], data_range[1], 200)
                    y = kde(x_range)
                    ax3.plot(x_range, y, color=color, linewidth=2, label=str(category))
                    mean_val = category_data.mean()
                    ax3.axvline(x=mean_val, color=color, linestyle='--', alpha=0.5)
                except:
                    continue
        
        ax3.set_xlabel(analysis_field, fontsize=11, fontname=font_name)
        ax3.set_ylabel('密度', fontsize=11, fontname=font_name)
        ax3.set_title(f'核密度估计图: 按 {category_field} 分组的 {analysis_field} 分布', 
                     fontsize=12, fontweight='bold', fontname=font_name)
        if len(categories) <= 8:
            ax3.legend(title=category_field, fontsize=9, title_fontsize=10)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '核密度估计图仅适用于数值类型数据', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=10, fontname=font_name)
        ax3.set_title(f'核密度估计图: 按 {category_field} 分组的 {analysis_field} 分布', 
                     fontsize=12, fontweight='bold', fontname=font_name)
    
    # ==========散点图绘制==========
    ax4 = fig.add_subplot(gs[1, 1])
    numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()
    hidden_field = None
    for field in numeric_fields:
        if field != category_field and field != analysis_field and field in df.columns:
            hidden_field = field
            break
    
    if hidden_field and hidden_field in df.columns:
        scatter_data = df[[category_field, analysis_field, hidden_field]].dropna().copy()
        
        if len(scatter_data) > 0:
            top_cat_scatter = scatter_data[category_field].value_counts().head(8).index
            scatter_data = scatter_data[scatter_data[category_field].isin(top_cat_scatter)].copy()
            scatter_categories = scatter_data[category_field].unique()
            
            if len(scatter_categories) > 0:
                colors = plt.cm.tab20(np.linspace(0, 1, len(scatter_categories)))
                markers = ['o', 's', '^', 'v', '<', '>', 'p', '*']
                
                for i, (category, color) in enumerate(zip(scatter_categories, colors)):
                    cat_data = scatter_data[scatter_data[category_field] == category]
                    marker = markers[i % len(markers)]
                    ax4.scatter(cat_data[hidden_field], cat_data[analysis_field],
                              c=[color], alpha=0.6, s=30, marker=marker,
                              label=str(category), edgecolor='black')
                
                ax4.set_xlabel(hidden_field, fontsize=11, fontname=font_name)
                ax4.set_ylabel(analysis_field, fontsize=11, fontname=font_name)
                ax4.set_title(f'散点图: {analysis_field} vs {hidden_field} (按 {category_field} 分组)', 
                             fontsize=12, fontweight='bold', fontname=font_name)
                
                if len(scatter_categories) <= 8:
                    ax4.legend(title=category_field, fontsize=8, title_fontsize=9)
                
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '无有效类别数据', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=10, fontname=font_name)
                ax4.set_title('散点图', fontsize=12, fontweight='bold', fontname=font_name)
        else:
            ax4.text(0.5, 0.5, '无有效数据用于散点图', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=10, fontname=font_name)
            ax4.set_title('散点图', fontsize=12, fontweight='bold', fontname=font_name)
    else:
        ax4.text(0.5, 0.5, '未找到合适的隐藏数值字段', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=10, fontname=font_name)
        ax4.set_title('散点图', fontsize=12, fontweight='bold', fontname=font_name)
    
    plt.suptitle(f'四个基础图表分析: 分类字段 = "{category_field}", 分析字段 = "{analysis_field}"', 
                fontsize=14, fontweight='bold', fontname=font_name, y=0.98)
    
    plt.tight_layout()
    return fig















# ==========主程序入口==========
def main():
    df = load_data()
    if df is None:
        print("无法加载数据，请检查文件路径")
        return
    
    print(f"数据加载成功！共{len(df)}行，{len(df.columns)}列")
    
    try:
        font_name = set_font()
        print(f"成功使用字体: {font_name}")
    except Exception as e:
        print(f"错误: {e}")
        print("请安装Noto Sans CJK JP字体：")
        print("1. 下载字体: https://fonts.google.com/noto/specimen/Noto+Sans+JP")
        print("2. 安装后重启程序")
        return
    
    all_fields = df.columns.tolist()
    initial_category = all_fields[0] if len(all_fields) > 0 else ""
    initial_analysis = all_fields[1] if len(all_fields) > 1 else initial_category
    
    initial_category_display, _, _ = get_display_fields(df, initial_category, "category", 0)
    initial_analysis_display, _, _ = get_display_fields(df, initial_analysis, "analysis", 0)


 
    # ==========Gradio界面构建==========
    with gr.Blocks(title="CSV数据可视化分析", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊📊 CSV数据可视化分析工具")
        gr.Markdown(f"**数据集信息**: {len(df)}行 × {len(df.columns)}列")
        gr.Markdown("### 选择分类字段和分析字段，探索它们之间的关系")
        gr.Markdown("**四个基础图表**: 1) 分组直方图 2) 占比饼图 3) 核密度估计图 4) 分组散点图")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🏷🏷🏷️ 分类字段选择")
                    category_scroll = gr.State(value=0)
                    
                    with gr.Row():
                        category_scroll_up = gr.Button("⬆⬆", variant="secondary", size="sm", scale=1)
                        category_scroll_down = gr.Button("⬇⬇", variant="secondary", size="sm", scale=1)
                    
                    category_field = gr.Dropdown(
                        choices=initial_category_display,
                        value=initial_category,
                        label="选择分类字段",
                        interactive=True,
                        allow_custom_value=False
                    )
                    
                    category_field_all = gr.Dropdown(
                        choices=all_fields,
                        value=initial_category,
                        label="或从全部字段中选择",
                        interactive=True,
                        allow_custom_value=False
                    )
                
                with gr.Group():
                    gr.Markdown("### 📈📈 分析字段选择")
                    analysis_scroll = gr.State(value=0)
                    
                    with gr.Row():
                        analysis_scroll_up = gr.Button("⬆⬆", variant="secondary", size="sm", scale=1)
                        analysis_scroll_down = gr.Button("⬇⬇", variant="secondary", size="sm", scale=1)
                    
                    analysis_field = gr.Dropdown(
                        choices=initial_analysis_display,
                        value=initial_analysis,
                        label="选择分析字段",
                        interactive=True,
                        allow_custom_value=False
                    )
                    
                    analysis_field_all = gr.Dropdown(
                        choices=all_fields,
                        value=initial_analysis,
                        label="或从全部字段中选择",
                        interactive=True,
                        allow_custom_value=False
                    )
                
                plot_btn = gr.Button("📊📊 生成四个图表", variant="primary", size="lg")
                
                with gr.Group():
                    gr.Markdown("### ℹℹ️ 字段信息")
                    field_info = gr.Markdown(f"""
                    **当前选择:**
                    - 分类字段: {initial_category}
                    - 分析字段: {initial_analysis}
                    
                    **数据集信息:**
                    - 总行数: {len(df)}
                    - 总列数: {len(df.columns)}
                    - 总字段数: {len(all_fields)}
                    """)
            
            with gr.Column(scale=2):
                output_plot = gr.Plot(label="四个基础图表", show_label=True)
        
        with gr.Accordion("📋📋 数据预览 (前5行)", open=False):
            data_preview = gr.DataFrame(value=df.head(), max_height=300, label="数据预览")








        
        # ==========回调函数定义==========
        def update_category_display(selected_field, scroll_offset):
            display_fields, new_selected, start_idx = get_display_fields(df, selected_field, "category", scroll_offset)
            return gr.Dropdown(choices=display_fields, value=new_selected), start_idx
        
        def update_analysis_display(selected_field, scroll_offset):
            display_fields, new_selected, start_idx = get_display_fields(df, selected_field, "analysis", scroll_offset)
            return gr.Dropdown(choices=display_fields, value=new_selected), start_idx
        
        def scroll_category_up(current_scroll, current_field):
            new_scroll = current_scroll - 1
            display_fields, new_selected, start_idx = get_display_fields(df, current_field, "category", new_scroll)
            return (gr.Dropdown(choices=display_fields, value=new_selected), 
                   new_scroll, 
                   new_selected)
        
        def scroll_category_down(current_scroll, current_field):
            new_scroll = current_scroll + 1
            display_fields, new_selected, start_idx = get_display_fields(df, current_field, "category", new_scroll)
            return (gr.Dropdown(choices=display_fields, value=new_selected), 
                   new_scroll, 
                   new_selected)
        
        def scroll_analysis_up(current_scroll, current_field):
            new_scroll = current_scroll - 1
            display_fields, new_selected, start_idx = get_display_fields(df, current_field, "analysis", new_scroll)
            return (gr.Dropdown(choices=display_fields, value=new_selected), 
                   new_scroll, 
                   new_selected)
        
        def scroll_analysis_down(current_scroll, current_field):
            new_scroll = current_scroll + 1
            display_fields, new_selected, start_idx = get_display_fields(df, current_field, "analysis", new_scroll)
            return (gr.Dropdown(choices=display_fields, value=new_selected), 
                   new_scroll, 
                   new_selected)
        
        def update_from_all_category(selected_field):
            display_fields, new_selected, start_idx = get_display_fields(df, selected_field, "category", 0)
            return (gr.Dropdown(choices=display_fields, value=new_selected), 
                   selected_field, 0)
        
        def update_from_all_analysis(selected_field):
            display_fields, new_selected, start_idx = get_display_fields(df, selected_field, "analysis", 0)
            return (gr.Dropdown(choices=display_fields, value=new_selected), 
                   selected_field, 0)
        
        def update_field_info(cat_field, ana_field):
            return f"""
            **当前选择:**
            - 分类字段: {cat_field}
            - 分析字段: {ana_field}
            
            **字段统计:**
            - 分类字段唯一值数量: {df[cat_field].nunique()}
            - 分析字段非空值数量: {df[ana_field].count()}
            - 分析字段唯一值数量: {df[ana_field].nunique()}
            
            **数据集信息:**
            - 总行数: {len(df)}
            - 总列数: {len(df.columns)}
            - 总字段数: {len(all_fields)}
            """
        
        def on_plot_click(cat_field, ana_field):
            fig = plot_four_basic_charts(df, cat_field, ana_field)
            info = update_field_info(cat_field, ana_field)
            return fig, info
        
        # ==========事件绑定==========
        category_scroll_up.click(
            fn=scroll_category_up,
            inputs=[category_scroll, category_field],
            outputs=[category_field, category_scroll, category_field_all]
        )
        
        category_scroll_down.click(
            fn=scroll_category_down,
            inputs=[category_scroll, category_field],
            outputs=[category_field, category_scroll, category_field_all]
        )
        
        analysis_scroll_up.click(
            fn=scroll_analysis_up,
            inputs=[analysis_scroll, analysis_field],
            outputs=[analysis_field, analysis_scroll, analysis_field_all]
        )
        
        analysis_scroll_down.click(
            fn=scroll_analysis_down,
            inputs=[analysis_scroll, analysis_field],
            outputs=[analysis_field, analysis_scroll, analysis_field_all]
        )
        
        category_field_all.change(
            fn=update_from_all_category,
            inputs=[category_field_all],
            outputs=[category_field, category_field_all, category_scroll]
        )
        
        analysis_field_all.change(
            fn=update_from_all_analysis,
            inputs=[analysis_field_all],
            outputs=[analysis_field, analysis_field_all, analysis_scroll]
        )
        
        category_field.change(
            fn=lambda x: (x, x, 0),
            inputs=[category_field],
            outputs=[category_field, category_field_all, category_scroll]
        )
        
        analysis_field.change(
            fn=lambda x: (x, x, 0),
            inputs=[analysis_field],
            outputs=[analysis_field, analysis_field_all, analysis_scroll]
        )
        
        plot_btn.click(
            fn=on_plot_click,
            inputs=[category_field, analysis_field],
            outputs=[output_plot, field_info]
        )
        
        demo.load(
            fn=lambda: (plot_four_basic_charts(df, initial_category, initial_analysis), 
                       update_field_info(initial_category, initial_analysis)),
            outputs=[output_plot, field_info]
        )
    
    # ==========应用启动==========
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)

if __name__ == "__main__":
    main()
