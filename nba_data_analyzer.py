import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 首先设置Seaborn的样式和调色板
sns.set_style("whitegrid")
sns.set_palette("coolwarm")

# 2. 然后，强制设置Matplotlib的字体，确保它不被覆盖
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10 # 设置一个全局字体大小，让图表更美观

class NBADataAnalyzer:
    """
    NBA数据可视化与分析器
    松耦合设计，通过读取预处理生成的文件进行分析
    """
    
    def __init__(self, processed_file_path: str, summary_file_path: str):
        """
        初始化分析器
        
        Args:
            processed_file_path: 预处理后的CSV文件路径
            summary_file_path: 预处理摘要JSON文件路径
        """
        self.processed_file_path = processed_file_path
        self.summary_file_path = summary_file_path
        self.data = None
        self.summary = None
        self.numeric_cols = None
        self.categorical_cols = None
        
        # 创建输出目录
        self.output_dir = "analysis_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("NBA数据可视化分析器初始化完成。")
        
    def load_data(self) -> bool:
        """加载预处理后的数据和摘要"""
        try:
            self.data = pd.read_csv(self.processed_file_path)
            print(f"成功加载处理后的数据: {self.data.shape[0]} 行, {self.data.shape[1]} 列")
            
            with open(self.summary_file_path, 'r', encoding='utf-8') as f:
                self.summary = json.load(f)
            print("成功加载处理摘要文件。")
            
            # 自动识别数值型和分类型列
            self._identify_column_types()
            
            return True
        except FileNotFoundError as e:
            print(f"错误: 文件未找到 - {e}")
            return False
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            return False
            
    def _identify_column_types(self):
        """根据数据类型和摘要信息识别数值型和分类型列"""
        if self.data is None:
            return
            
        # 从摘要中获取编码后的分类变量
        encoded_categoricals = self.summary.get('data_standardization', {}).get('categorical_encoding', [])
        
        # 获取所有数值列
        potential_numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        # 筛选出真正的连续变量（排除编码后的分类变量和一些ID）
        self.categorical_cols = list(set(encoded_categoricals + ['MONTH', 'IS_WEEKEND']))
        self.numeric_cols = [col for col in potential_numeric_cols if col not in self.categorical_cols]
        
        print(f"\n自动识别到数值型变量 ({len(self.numeric_cols)}个): {self.numeric_cols}")
        print(f"自动识别到分类型变量 ({len(self.categorical_cols)}个): {self.categorical_cols}")

    def run_all_analyses(self):
        """执行所有分析步骤"""
        if not self.load_data():
            return
            
        print("\n" + "="*50)
        print("开始执行完整的数据分析与可视化流程")
        print("="*50)
        
        # 5.3.1 数据初步分析
        self.initial_data_analysis()
        
        # 5.3.2 变量相关性分析
        self.correlation_analysis()
        
        # 5.3.3 特征选择
        self.feature_selection()
        
        # 5.3.4 特征提取
        self.feature_extraction()
        
        # 5.3.5 特征编码 (此步已在预处理中完成，此处进行验证和可视化)
        self.feature_encoding_analysis()
        
        print(f"\n所有分析完成！生成的图表已保存在 '{self.output_dir}' 文件夹中。")

    def _save_plot(self, fig, filename):
        """保存图表到文件"""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {filepath}")
        plt.close(fig)

    # ==================== 5.3.1 数据初步分析 ====================
    def initial_data_analysis(self):
        """数据初步分析：区分变量类型并可视化"""
        print("\n--- 5.3.1 数据初步分析 ---")
        
        # 1. 数值型变量分布分析
        if self.numeric_cols:
            # 选择一些关键的标准化后的数值变量进行绘图，避免图表过多
            key_numeric_cols = [col for col in self.numeric_cols if 'STANDARDIZED' in col][:6]
            if key_numeric_cols:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle('关键数值型变量分布图', fontsize=20)
                for i, col in enumerate(key_numeric_cols):
                    ax = axes[i//3, i%3]
                    sns.histplot(self.data[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f'{col} 分布')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                self._save_plot(fig, "5.3.1_numeric_distribution.png")

        # 2. 分类型变量计数分析
        if self.categorical_cols:
            fig, axes = plt.subplots(1, len(self.categorical_cols), figsize=(6 * len(self.categorical_cols), 5))
            if len(self.categorical_cols) == 1:
                axes = [axes]
            fig.suptitle('分类型变量计数图', fontsize=20)
            for i, col in enumerate(self.categorical_cols):
                if col in self.data.columns:
                    sns.countplot(x=col, data=self.data, ax=axes[i])
                    axes[i].set_title(f'{col} 计数')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self._save_plot(fig, "5.3.1_categorical_counts.png")

    # ==================== 5.3.2 变量相关性分析 ====================
    def correlation_analysis(self):
        """变量相关性分析（可视化）"""
        print("\n--- 5.3.2 变量相关性分析 ---")
        
        if not self.numeric_cols:
            print("没有足够的数值型变量进行相关性分析。")
            return

        # 计算相关性矩阵
        corr_matrix = self.data[self.numeric_cols].corr()
        
        # 绘制热力图
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('数值型变量相关性热力图', fontsize=20)
        self._save_plot(plt.gcf(), "5.3.2_correlation_heatmap.png")

    # ==================== 5.3.3 特征选择 ====================
    def feature_selection(self):
        """特征选择：使用随机森林评估特征重要性"""
        print("\n--- 5.3.3 特征选择 ---")
        
        # 使用比赛结果(WL_ENCODED)作为目标变量
        target_col = 'WL_ENCODED'
        if target_col not in self.data.columns:
            print(f"目标变量 '{target_col}' 未找到，无法进行特征选择。")
            return
            
        X = self.data[self.numeric_cols].fillna(0)
        y = self.data[target_col]
        
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
        plt.title('Top 15 特征重要性 (基于随机森林)', fontsize=20)
        plt.tight_layout()
        self._save_plot(plt.gcf(), "5.3.3_feature_importance.png")

    # ==================== 5.3.4 特征提取 ====================
    def feature_extraction(self):
        """特征提取：使用PCA进行降维"""
        print("\n--- 5.3.4 特征提取 (PCA) ---")
        
        if not self.numeric_cols:
            print("没有足够的数值型变量进行PCA。")
            return
            
        # 只对标准化后的数据进行PCA
        standardized_cols = [col for col in self.numeric_cols if 'STANDARDIZED' in col]
        if not standardized_cols:
            print("未找到标准化后的变量，PCA效果可能不佳。")
            standardized_cols = self.numeric_cols

        X = self.data[standardized_cols].fillna(0)
        
        # 使用PCA降维到2个主成分
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        
        # 可视化主成分
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6)
        plt.title('PCA降维结果 (2个主成分)', fontsize=20)
        plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        self._save_plot(plt.gcf(), "5.3.4_pca_visualization.png")
        
        print(f"PCA完成，前两个主成分共解释了 {pca.explained_variance_ratio_.sum():.2%} 的方差。")

    # ==================== 5.3.5 特征编码 ====================
    def feature_encoding_analysis(self):
        """特征编码分析：验证并可视化编码效果"""
        print("\n--- 5.3.5 特征编码分析 ---")
        
        # 检查编码后的分类变量
        encoded_cols = self.summary.get('data_standardization', {}).get('categorical_encoding', [])
        
        if not encoded_cols or not any(col in self.data.columns for col in encoded_cols):
            print("未找到有效的编码后变量。")
            return
            
        # 以WL_ENCODED为例，查看其与原始变量的关系（如果原始变量还在）
        # 由于processed文件只包含新列，我们直接展示编码后的分布
        if 'WL_ENCODED' in self.data.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x='WL_ENCODED', data=self.data)
            plt.title('比赛结果 (WL_ENCODED) 分布', fontsize=16)
            plt.xlabel('编码后的比赛结果 (0: 负, 1: 胜)')
            plt.ylabel('场次')
            self._save_plot(plt.gcf(), "5.3.5_wl_encoded_distribution.png")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 假设你的文件在processed_data文件夹中
    # 请根据你的实际文件名修改路径
    processed_file = './processed_data/regular_season_totals_2010_2024_processed.csv'
    summary_file = './processed_data/regular_season_totals_2010_2024_summary.json'
    
    # 检查文件是否存在
    if not os.path.exists(processed_file) or not os.path.exists(summary_file):
        print("错误：请确保processed文件和summary文件存在于指定路径。")
        print(f"查找文件: {processed_file}")
        print(f"查找文件: {summary_file}")
    else:
        # 创建分析器实例并运行所有分析
        analyzer = NBADataAnalyzer(processed_file, summary_file)
        analyzer.run_all_analyses()

