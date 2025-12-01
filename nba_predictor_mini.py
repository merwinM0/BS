import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NBAGamePredictorMini:
    """
    NBA比赛结果预测器 - Mini版本
    使用采样数据和轻量级模型快速测试
    """
    
    def __init__(self, processed_file: str, summary_file: str, original_file: str = None,
                 sample_size: int = 2000):
        """
        初始化预测器
        
        Args:
            processed_file: 预处理后的CSV文件路径
            summary_file: 预处理摘要JSON文件路径
            original_file: 原始数据文件路径（可选）
            sample_size: 采样数量，默认2000条
        """
        self.processed_file = processed_file
        self.summary_file = summary_file
        self.original_file = original_file
        self.sample_size = sample_size
        
        self.data = None
        self.summary = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        
        # 输出目录
        self.output_dir = "prediction_output_mini"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"NBA比赛结果预测器(Mini版)初始化完成 - 采样{sample_size}条数据")
    
    def load_and_prepare_data(self) -> bool:
        """加载并准备数据（采样）"""
        try:
            # 加载预处理后的数据
            processed_data = pd.read_csv(self.processed_file)
            print(f"加载预处理数据: {processed_data.shape}")
            
            # 随机采样
            if len(processed_data) > self.sample_size:
                processed_data = processed_data.sample(n=self.sample_size, random_state=42)
                print(f"采样后: {processed_data.shape}")
            
            # 加载摘要
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                self.summary = json.load(f)
            
            # 如果有原始文件，合并数据
            if self.original_file and os.path.exists(self.original_file):
                original_data = pd.read_csv(self.original_file)
                print(f"加载原始数据: {original_data.shape}")
                
                # 同样采样
                if len(original_data) > self.sample_size:
                    original_data = original_data.sample(n=self.sample_size, random_state=42)
                
                # 合并数据
                self.data = pd.concat([original_data.reset_index(drop=True), 
                                      processed_data.reset_index(drop=True)], axis=1)
                self.data = self.data.loc[:, ~self.data.columns.duplicated()]
            else:
                self.data = processed_data
            
            print(f"最终数据形状: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def prepare_features(self, feature_cols: list = None, target_col: str = 'WL_ENCODED'):
        """准备特征和目标变量"""
        if self.data is None:
            print("请先加载数据")
            return
        
        # 确保目标变量存在
        if target_col not in self.data.columns:
            if 'WL' in self.data.columns:
                self.data['WL_ENCODED'] = (self.data['WL'] == 'W').astype(int)
            else:
                print(f"目标变量 {target_col} 不存在")
                return
        
        # 数据泄露特征（直接包含比赛结果信息，必须排除）
        leakage_features = ['PLUS_MINUS', 'PLUS_MINUS_STANDARDIZED', 'PLUS_MINUS_NORMALIZED',
                           'WL', 'WL_ENCODED', 'W', 'L']
        
        # 自动选择特征
        if feature_cols is None:
            standardized_cols = [col for col in self.data.columns if 'STANDARDIZED' in col]
            normalized_cols = [col for col in self.data.columns if 'NORMALIZED' in col]
            derived_cols = ['FG_EFFICIENCY', 'TRUE_SHOOTING_PCT', 'OFFENSIVE_RATING', 
                           'OPPONENT_STRENGTH_SCORE']
            derived_cols = [col for col in derived_cols if col in self.data.columns]
            time_cols = ['MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND']
            time_cols = [col for col in time_cols if col in self.data.columns]
            
            feature_cols = standardized_cols + derived_cols + time_cols
            
            if len(feature_cols) < 5:
                numeric_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                               'TOV', 'STL', 'BLK']
                feature_cols = [col for col in numeric_cols if col in self.data.columns]
            
            # 移除泄露特征
            feature_cols = [col for col in feature_cols if col not in leakage_features]
        
        self.feature_names = feature_cols
        print(f"\n使用的特征 ({len(feature_cols)}个): {feature_cols[:5]}...")  # 只显示前5个
        
        # 准备X和y
        X = self.data[feature_cols].copy()
        y = self.data[target_col].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 删除目标变量为空的行
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {len(self.X_train)}")
        print(f"测试集大小: {len(self.X_test)}")
        print(f"正样本比例: {y.mean():.2%}")
    
    def train_models(self):
        """训练轻量级模型（不含SVM）"""
        if self.X_train is None:
            print("请先准备特征数据")
            return
        
        print("\n" + "="*50)
        print("开始训练模型 (Mini版 - 轻量级配置)")
        print("="*50)
        
        # 轻量级模型配置（不含SVM）
        models_config = {
            '逻辑回归': LogisticRegression(max_iter=500, random_state=42),
            '随机森林': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            '梯度提升': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'K近邻': KNeighborsClassifier(n_neighbors=5),
        }
        
        for name, model in models_config.items():
            print(f"\n训练 {name}...")
            
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            result = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            if y_prob is not None:
                result['auc'] = roc_auc_score(self.y_test, y_prob)
            
            self.results[name] = result
            
            print(f"  准确率: {result['accuracy']:.4f}")
            print(f"  F1分数: {result['f1']:.4f}")
            if 'auc' in result:
                print(f"  AUC: {result['auc']:.4f}")
    
    def cross_validation(self, cv=3):
        """交叉验证（3折，更快）"""
        if self.X_train is None:
            return
        
        print(f"\n{cv}折交叉验证")
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    def plot_results(self):
        """绘制结果图"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 1. 准确率对比
        names = list(self.results.keys())
        accuracies = [self.results[n]['accuracy'] for n in names]
        axes[0].barh(names, accuracies, color='steelblue')
        axes[0].set_xlabel('准确率')
        axes[0].set_title('模型准确率对比')
        axes[0].set_xlim(0, 1)
        
        # 2. ROC曲线
        for name, result in self.results.items():
            if result['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_prob'])
                roc_auc = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlabel('FPR')
        axes[1].set_ylabel('TPR')
        axes[1].set_title('ROC曲线')
        axes[1].legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'mini_results.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n结果图已保存: {filepath}")
        plt.close()
    
    def generate_report(self):
        """生成简要报告"""
        if not self.results:
            return
        
        print("\n" + "="*50)
        print("预测结果摘要")
        print("="*50)
        print(f"{'模型':<10} {'准确率':<10} {'F1':<10} {'AUC':<10}")
        print("-"*40)
        
        for name, result in self.results.items():
            auc_val = result.get('auc', 'N/A')
            auc_str = f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val
            print(f"{name:<10} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {auc_str:<10}")
        
        best = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n最佳模型: {best[0]} (准确率: {best[1]['accuracy']:.4f})")
    
    def run(self):
        """运行Mini预测流程"""
        print("\n" + "="*60)
        print(f"NBA比赛结果预测 - Mini版 (采样{self.sample_size}条)")
        print("="*60)
        
        if not self.load_and_prepare_data():
            return
        
        self.prepare_features()
        self.train_models()
        self.cross_validation()
        self.plot_results()
        self.generate_report()
        
        print("\n" + "="*60)
        print("Mini预测流程完成！")
        print("="*60)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    processed_file = './processed_data/regular_season_totals_2010_2024_processed.csv'
    summary_file = './processed_data/regular_season_totals_2010_2024_summary.json'
    original_file = './NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv'
    
    if not os.path.exists(processed_file):
        print(f"错误: 找不到预处理文件 {processed_file}")
        print("请先运行 data_preprocessor.py 进行数据预处理")
    else:
        # 创建Mini预测器并运行（默认采样2000条）
        predictor = NBAGamePredictorMini(processed_file, summary_file, original_file, 
                                         sample_size=2000)
        predictor.run()
