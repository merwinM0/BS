"""
NBA比赛结果预测系统
- 训练并保存模型
- 加载模型进行预测
- 输入两支球队，预测胜负
- 可用球队ATL, BKN, BOS, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU,
IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, 
PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS
"""


import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class NBAModelPredictor:
    """NBA比赛预测器 - 支持模型保存和加载"""
    
    def __init__(self, model_dir: str = "saved_model"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.team_stats = None  # 球队历史统计数据
        
    def load_data(self, processed_file: str, original_file: str = None):
        """加载数据"""
        processed_data = pd.read_csv(processed_file)
        print(f"加载数据: {processed_data.shape}")
        
        if original_file and os.path.exists(original_file):
            original_data = pd.read_csv(original_file)
            # 合并数据
            self.data = pd.concat([original_data, processed_data], axis=1)
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        else:
            self.data = processed_data
        
        # 计算每支球队的历史平均统计数据
        self._compute_team_stats()
        
        return self
    
    def _compute_team_stats(self):
        """计算每支球队的历史平均统计数据"""
        # 找到球队标识列
        team_col = None
        for col in ['TEAM_ABBREVIATION', 'TEAM_NAME', 'TEAM']:
            if col in self.data.columns:
                team_col = col
                break
        
        if team_col is None:
            print("警告: 未找到球队标识列")
            return
        
        # 统计特征
        stat_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                    'TOV', 'STL', 'BLK']
        stat_cols = [col for col in stat_cols if col in self.data.columns]
        
        # 计算每支球队的平均数据
        self.team_stats = self.data.groupby(team_col)[stat_cols].mean()
        self.team_col = team_col
        
        print(f"已计算 {len(self.team_stats)} 支球队的历史统计数据")
        print(f"球队列表: {list(self.team_stats.index)}")
    
    def prepare_features(self):
        """准备训练特征"""
        # 泄露特征
        leakage_features = ['PLUS_MINUS', 'PLUS_MINUS_STANDARDIZED', 'PLUS_MINUS_NORMALIZED',
                           'WL', 'WL_ENCODED', 'W', 'L']
        
        # 选择数值特征
        feature_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                       'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                       'TOV', 'STL', 'BLK']
        feature_cols = [col for col in feature_cols if col in self.data.columns]
        feature_cols = [col for col in feature_cols if col not in leakage_features]
        
        self.feature_names = feature_cols
        print(f"使用特征: {feature_cols}")
        
        # 准备目标变量
        if 'WL_ENCODED' not in self.data.columns:
            if 'WL' in self.data.columns:
                self.data['WL_ENCODED'] = (self.data['WL'] == 'W').astype(int)
        
        X = self.data[feature_cols].fillna(self.data[feature_cols].mean())
        y = self.data['WL_ENCODED']
        
        # 删除无效行
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def train(self, test_size: float = 0.2):
        """训练模型"""
        print("\n" + "="*50)
        print("开始训练模型")
        print("="*50)
        
        X, y = self.prepare_features()
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练逻辑回归模型（表现最好）
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # 评估
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"模型准确率: {accuracy:.4f}")
        
        return accuracy
    
    def save_model(self):
        """保存模型到文件"""
        if self.model is None:
            print("请先训练模型")
            return
        
        # 保存模型
        model_path = os.path.join(self.model_dir, "nba_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存特征名
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'feature_names': self.feature_names,
                'team_col': self.team_col
            }, f, ensure_ascii=False, indent=2)
        
        # 保存球队统计数据
        stats_path = os.path.join(self.model_dir, "team_stats.csv")
        self.team_stats.to_csv(stats_path)
        
        print(f"\n模型已保存到 '{self.model_dir}' 文件夹")
        print(f"  - nba_model.pkl: 预测模型")
        print(f"  - scaler.pkl: 数据标准化器")
        print(f"  - config.json: 配置信息")
        print(f"  - team_stats.csv: 球队历史数据")
    
    def load_model(self):
        """从文件加载模型"""
        model_path = os.path.join(self.model_dir, "nba_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        config_path = os.path.join(self.model_dir, "config.json")
        stats_path = os.path.join(self.model_dir, "team_stats.csv")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.feature_names = config['feature_names']
            self.team_col = config['team_col']
        
        self.team_stats = pd.read_csv(stats_path, index_col=0)
        
        print("模型加载成功!")
        return True
    
    def get_team_list(self):
        """获取可用的球队列表"""
        if self.team_stats is None:
            return []
        return list(self.team_stats.index)
    
    def predict(self, team1: str, team2: str):
        """
        预测两支球队比赛结果
        
        Args:
            team1: 球队1名称/缩写
            team2: 球队2名称/缩写
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            print("请先加载或训练模型")
            return None
        
        # 查找球队
        team1_upper = team1.upper()
        team2_upper = team2.upper()
        
        teams = self.get_team_list()
        
        # 模糊匹配
        team1_match = None
        team2_match = None
        
        for t in teams:
            if team1_upper in t.upper() or t.upper() in team1_upper:
                team1_match = t
            if team2_upper in t.upper() or t.upper() in team2_upper:
                team2_match = t
        
        if team1_match is None:
            print(f"未找到球队: {team1}")
            print(f"可用球队: {teams}")
            return None
        
        if team2_match is None:
            print(f"未找到球队: {team2}")
            print(f"可用球队: {teams}")
            return None
        
        # 获取球队统计数据
        stats1 = self.team_stats.loc[team1_match][self.feature_names].values
        stats2 = self.team_stats.loc[team2_match][self.feature_names].values
        
        # 标准化
        stats1_scaled = self.scaler.transform([stats1])
        stats2_scaled = self.scaler.transform([stats2])
        
        # 预测
        prob1 = self.model.predict_proba(stats1_scaled)[0][1]  # team1胜率
        prob2 = self.model.predict_proba(stats2_scaled)[0][1]  # team2胜率
        
        # 计算相对胜率
        total = prob1 + prob2
        team1_win_prob = prob1 / total
        team2_win_prob = prob2 / total
        
        # 结果
        result = {
            'team1': team1_match,
            'team2': team2_match,
            'team1_win_prob': team1_win_prob,
            'team2_win_prob': team2_win_prob,
            'predicted_winner': team1_match if team1_win_prob > team2_win_prob else team2_match
        }
        
        return result
    
    def predict_interactive(self):
        """交互式预测"""
        print("\n" + "="*50)
        print("NBA比赛结果预测系统")
        print("="*50)
        print(f"可用球队: {self.get_team_list()}")
        print("输入 'quit' 退出\n")
        
        while True:
            team1 = input("请输入球队1 (如 LAL, GSW): ").strip()
            if team1.lower() == 'quit':
                break
            
            team2 = input("请输入球队2 (如 BOS, MIA): ").strip()
            if team2.lower() == 'quit':
                break
            
            result = self.predict(team1, team2)
            
            if result:
                print("\n" + "-"*40)
                print(f"🏀 {result['team1']} vs {result['team2']}")
                print(f"   {result['team1']} 胜率: {result['team1_win_prob']:.1%}")
                print(f"   {result['team2']} 胜率: {result['team2_win_prob']:.1%}")
                print(f"   预测获胜: {result['predicted_winner']} 🏆")
                print("-"*40 + "\n")


def train_and_save():
    """训练并保存模型"""
    processed_file = './processed_data/regular_season_totals_2010_2024_processed.csv'
    original_file = './NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv'
    
    if not os.path.exists(processed_file):
        print(f"错误: 找不到文件 {processed_file}")
        return
    
    predictor = NBAModelPredictor()
    predictor.load_data(processed_file, original_file)
    predictor.train()
    predictor.save_model()
    
    return predictor


def load_and_predict():
    """加载模型并进行预测"""
    predictor = NBAModelPredictor()
    
    if not predictor.load_model():
        print("模型不存在，开始训练...")
        predictor = train_and_save()
    
    return predictor


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import sys
    
    # 检查是否有命令行参数
    if len(sys.argv) >= 3:
        # 命令行模式: python nba_model_predictor.py LAL GSW
        predictor = load_and_predict()
        result = predictor.predict(sys.argv[1], sys.argv[2])
        if result:
            print(f"\n{result['team1']} vs {result['team2']}")
            print(f"{result['team1']} 胜率: {result['team1_win_prob']:.1%}")
            print(f"{result['team2']} 胜率: {result['team2_win_prob']:.1%}")
            print(f"预测获胜: {result['predicted_winner']}")
    else:
        # 交互模式
        predictor = load_and_predict()
        predictor.predict_interactive()
