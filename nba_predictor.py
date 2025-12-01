import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NBAGamePredictor:
    """
    NBA比赛结果预测器
    使用多种机器学习模型预测比赛胜负
    """
    
    def __init__(self, processed_file: str, summary_file: str, original_file: str = None):
        """
        初始化预测器
        
        Args:
            processed_file: 预处理后的CSV文件路径
            summary_file: 预处理摘要JSON文件路径
            original_file: 原始数据文件路径（可选，用于合并完整数据）
        """
        self.processed_file = processed_file
        self.summary_file = summary_file
        self.original_file = original_file
        
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
        self.output_dir = "prediction_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("NBA比赛结果预测器初始化完成")
    
    def load_and_prepare_data(self) -> bool:
        """加载并准备数据"""
        try:
            # 加载预处理后的数据
            processed_data = pd.read_csv(self.processed_file)
            print(f"加载预处理数据: {processed_data.shape}")
            
            # 加载摘要
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                self.summary = json.load(f)
            
            # 如果有原始文件，合并数据以获取更多特征
            if self.original_file and os.path.exists(self.original_file):
                original_data = pd.read_csv(self.original_file)
                print(f"加载原始数据: {original_data.shape}")
                
                # 合并数据（按索引）
                self.data = pd.concat([original_data, processed_data], axis=1)
                # 删除重复列
                self.data = self.data.loc[:, ~self.data.columns.duplicated()]
            else:
                self.data = processed_data
            
            print(f"最终数据形状: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def prepare_features(self, feature_cols: list = None, target_col: str = 'WL_ENCODED'):
        """
        准备特征和目标变量
        
        Args:
            feature_cols: 特征列列表，None则自动选择
            target_col: 目标变量列名
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 确保目标变量存在
        if target_col not in self.data.columns:
            # 如果没有编码，手动编码
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
            # 优先使用标准化后的特征
            standardized_cols = [col for col in self.data.columns if 'STANDARDIZED' in col]
            normalized_cols = [col for col in self.data.columns if 'NORMALIZED' in col]
            
            # 衍生特征
            derived_cols = ['FG_EFFICIENCY', 'TRUE_SHOOTING_PCT', 'OFFENSIVE_RATING', 
                           'OPPONENT_STRENGTH_SCORE']
            derived_cols = [col for col in derived_cols if col in self.data.columns]
            
            # 时间特征
            time_cols = ['MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND']
            time_cols = [col for col in time_cols if col in self.data.columns]
            
            # 组合特征
            feature_cols = standardized_cols + derived_cols + time_cols
            
            # 如果标准化特征不够，使用原始数值特征
            if len(feature_cols) < 5:
                numeric_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                               'TOV', 'STL', 'BLK']
                feature_cols = [col for col in numeric_cols if col in self.data.columns]
            
            # 移除泄露特征
            feature_cols = [col for col in feature_cols if col not in leakage_features]
        
        self.feature_names = feature_cols
        print(f"\n使用的特征 ({len(feature_cols)}个): {feature_cols}")
        
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
        """训练多个模型"""
        if self.X_train is None:
            print("请先准备特征数据")
            return
        
        print("\n" + "="*50)
        print("开始训练模型")
        print("="*50)
        
        # 定义模型
        models_config = {
            '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'K近邻': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        for name, model in models_config.items():
            print(f"\n训练 {name}...")
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            # 预测
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 评估
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
            print(f"  精确率: {result['precision']:.4f}")
            print(f"  召回率: {result['recall']:.4f}")
            print(f"  F1分数: {result['f1']:.4f}")
            if 'auc' in result:
                print(f"  AUC: {result['auc']:.4f}")
    
    def cross_validation(self, cv=5):
        """交叉验证评估"""
        if self.X_train is None:
            print("请先准备特征数据")
            return
        
        print("\n" + "="*50)
        print(f"{cv}折交叉验证")
        print("="*50)
        
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        
        cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return cv_results
    
    def plot_model_comparison(self):
        """绘制模型比较图"""
        if not self.results:
            print("请先训练模型")
            return
        
        # 准备数据
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 柱状图比较各指标
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            axes[0].bar(x + i*width, values, width, label=metric.upper())
        
        axes[0].set_xlabel('模型')
        axes[0].set_ylabel('分数')
        axes[0].set_title('模型性能比较')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(model_names, rotation=15)
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        # 2. ROC曲线
        for name, result in self.results.items():
            if result['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_prob'])
                roc_auc = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='随机猜测')
        axes[1].set_xlabel('假正率 (FPR)')
        axes[1].set_ylabel('真正率 (TPR)')
        axes[1].set_title('ROC曲线比较')
        axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n模型比较图已保存: {filepath}")
        plt.close()
    
    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        if not self.results:
            print("请先训练模型")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['负', '胜'], yticklabels=['负', '胜'])
            axes[i].set_title(f'{name}\n准确率: {result["accuracy"]:.4f}')
            axes[i].set_xlabel('预测')
            axes[i].set_ylabel('实际')
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存: {filepath}")
        plt.close()
    
    def plot_feature_importance(self):
        """绘制特征重要性（针对树模型）"""
        if '随机森林' not in self.models:
            print("随机森林模型未训练")
            return
        
        rf_model = self.models['随机森林']
        importances = rf_model.feature_importances_
        
        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('重要性')
        plt.title('特征重要性 (随机森林)')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_importance_prediction.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存: {filepath}")
        plt.close()
    
    def hyperparameter_tuning(self, model_name: str = '随机森林'):
        """超参数调优"""
        if self.X_train is None:
            print("请先准备特征数据")
            return
        
        print(f"\n开始 {model_name} 超参数调优...")
        
        if model_name == '随机森林':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == '逻辑回归':
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            print(f"暂不支持 {model_name} 的超参数调优")
            return
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 使用最佳模型预测
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        return grid_search.best_estimator_
    
    def generate_report(self):
        """生成预测报告"""
        if not self.results:
            print("请先训练模型")
            return
        
        report = []
        report.append("="*60)
        report.append("NBA比赛结果预测报告")
        report.append("="*60)
        report.append(f"\n数据集信息:")
        report.append(f"  - 总样本数: {len(self.data)}")
        report.append(f"  - 训练集大小: {len(self.X_train)}")
        report.append(f"  - 测试集大小: {len(self.X_test)}")
        report.append(f"  - 特征数量: {len(self.feature_names)}")
        
        report.append(f"\n使用的特征:")
        for feat in self.feature_names:
            report.append(f"  - {feat}")
        
        report.append(f"\n模型性能对比:")
        report.append("-"*60)
        report.append(f"{'模型':<12} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'AUC':<10}")
        report.append("-"*60)
        
        for name, result in self.results.items():
            auc_val = result.get('auc', 'N/A')
            auc_str = f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val
            report.append(f"{name:<12} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                         f"{result['recall']:<10.4f} {result['f1']:<10.4f} {auc_str:<10}")
        
        report.append("-"*60)
        
        # 找出最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.4f})")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 保存报告
        filepath = os.path.join(self.output_dir, 'prediction_report.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存: {filepath}")
    
    def run_full_pipeline(self):
        """运行完整的预测流程"""
        print("\n" + "="*60)
        print("NBA比赛结果预测 - 完整流程")
        print("="*60)
        
        # 1. 加载数据
        if not self.load_and_prepare_data():
            return
        
        # 2. 准备特征
        self.prepare_features()
        
        # 3. 训练模型
        self.train_models()
        
        # 4. 交叉验证
        self.cross_validation()
        
        # 5. 可视化
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        
        # 6. 生成报告
        self.generate_report()
        
        print("\n" + "="*60)
        print("预测流程完成！")
        print(f"所有结果已保存到 '{self.output_dir}' 文件夹")
        print("="*60)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 文件路径
    processed_file = './processed_data/regular_season_totals_2010_2024_processed.csv'
    summary_file = './processed_data/regular_season_totals_2010_2024_summary.json'
    original_file = './NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv'
    
    # 检查文件
    if not os.path.exists(processed_file):
        print(f"错误: 找不到预处理文件 {processed_file}")
        print("请先运行 data_preprocessor.py 进行数据预处理")
    else:
        # 创建预测器并运行
        predictor = NBAGamePredictor(processed_file, summary_file, original_file)
        predictor.run_full_pipeline()
        
        # 可选：超参数调优
        # best_rf = predictor.hyperparameter_tuning('随机森林')
