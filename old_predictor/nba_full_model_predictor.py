"""
NBA比赛结果预测系统 - 完整版
使用所有可用数据进行训练：
- 常规赛球队统计数据 (regular_season_totals)
- 季后赛球队统计数据 (play_off_totals)
- 常规赛球员详细数据 (regular_season_box_scores)
- 季后赛球员详细数据 (play_off_box_scores)

可用球队: ATL, BKN, BOS, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU,
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class NBAFullModelPredictor:
    """NBA比赛预测器 - 使用所有数据训练"""
    
    def __init__(self, model_dir: str = "saved_model_full"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.team_stats = None
        self.team_col = 'TEAM_ABBREVIATION'
        
        # 数据文件路径
        self.data_dir = './NBA-Data-2010-2024-main'
        
    def load_all_data(self):
        """加载所有可用数据"""
        print("="*60)
        print("加载所有NBA数据...")
        print("="*60)
        
        all_team_data = []
        all_player_data = []
        
        # 1. 加载常规赛球队统计
        regular_totals_file = os.path.join(self.data_dir, 'regular_season_totals_2010_2024.csv')
        if os.path.exists(regular_totals_file):
            df = pd.read_csv(regular_totals_file)
            df['DATA_SOURCE'] = 'regular_season'
            all_team_data.append(df)
            print(f"✓ 常规赛球队统计: {len(df)} 条记录")
        
        # 2. 加载季后赛球队统计
        playoff_totals_file = os.path.join(self.data_dir, 'play_off_totals_2010_2024.csv')
        if os.path.exists(playoff_totals_file):
            df = pd.read_csv(playoff_totals_file)
            df['DATA_SOURCE'] = 'playoff'
            all_team_data.append(df)
            print(f"✓ 季后赛球队统计: {len(df)} 条记录")
        
        # 3. 加载常规赛球员详细数据
        for i in range(1, 4):
            box_file = os.path.join(self.data_dir, f'regular_season_box_scores_2010_2024_part_{i}.csv')
            if os.path.exists(box_file):
                df = pd.read_csv(box_file)
                df['DATA_SOURCE'] = 'regular_season_box'
                all_player_data.append(df)
                print(f"✓ 常规赛球员数据 Part {i}: {len(df)} 条记录")
        
        # 4. 加载季后赛球员详细数据
        playoff_box_file = os.path.join(self.data_dir, 'play_off_box_scores_2010_2024.csv')
        if os.path.exists(playoff_box_file):
            df = pd.read_csv(playoff_box_file)
            df['DATA_SOURCE'] = 'playoff_box'
            all_player_data.append(df)
            print(f"✓ 季后赛球员数据: {len(df)} 条记录")
        
        # 合并球队数据
        if all_team_data:
            self.team_data = pd.concat(all_team_data, ignore_index=True)
            print(f"\n球队数据总计: {len(self.team_data)} 条记录")
        
        # 合并球员数据
        if all_player_data:
            self.player_data = pd.concat(all_player_data, ignore_index=True)
            print(f"球员数据总计: {len(self.player_data)} 条记录")
        
        # 计算球队统计
        self._compute_team_stats()
        self._compute_player_aggregated_stats()
        
        return self
    
    def _compute_team_stats(self):
        """计算每支球队的历史平均统计数据"""
        print("\n计算球队历史统计...")
        
        # 统计特征
        stat_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                    'TOV', 'STL', 'BLK', 'PLUS_MINUS']
        stat_cols = [col for col in stat_cols if col in self.team_data.columns]
        
        # 计算每支球队的平均数据
        self.team_stats = self.team_data.groupby('TEAM_ABBREVIATION')[stat_cols].mean()
        
        # 计算胜率
        if 'WL' in self.team_data.columns:
            win_rate = self.team_data.groupby('TEAM_ABBREVIATION')['WL'].apply(
                lambda x: (x == 'W').sum() / len(x)
            )
            self.team_stats['WIN_RATE'] = win_rate
        
        # 计算常规赛和季后赛分开的统计
        regular_data = self.team_data[self.team_data['DATA_SOURCE'] == 'regular_season']
        playoff_data = self.team_data[self.team_data['DATA_SOURCE'] == 'playoff']
        
        if len(regular_data) > 0:
            regular_stats = regular_data.groupby('TEAM_ABBREVIATION')[stat_cols].mean()
            regular_stats.columns = [f'{col}_REGULAR' for col in regular_stats.columns]
            self.team_stats = self.team_stats.join(regular_stats)
        
        if len(playoff_data) > 0:
            playoff_stats = playoff_data.groupby('TEAM_ABBREVIATION')[stat_cols].mean()
            playoff_stats.columns = [f'{col}_PLAYOFF' for col in playoff_stats.columns]
            self.team_stats = self.team_stats.join(playoff_stats)
        
        print(f"✓ 已计算 {len(self.team_stats)} 支球队的历史统计数据")
        print(f"  球队列表: {list(self.team_stats.index)}")
    
    def _compute_player_aggregated_stats(self):
        """从球员数据中聚合球队统计"""
        print("\n从球员数据聚合球队统计...")
        
        if not hasattr(self, 'player_data') or self.player_data is None:
            return
        
        # 球员数据的列名映射
        col_mapping = {
            'teamTricode': 'TEAM_ABBREVIATION',
            'points': 'PTS_PLAYER',
            'fieldGoalsMade': 'FGM_PLAYER',
            'fieldGoalsAttempted': 'FGA_PLAYER',
            'fieldGoalsPercentage': 'FG_PCT_PLAYER',
            'threePointersMade': 'FG3M_PLAYER',
            'threePointersAttempted': 'FG3A_PLAYER',
            'threePointersPercentage': 'FG3_PCT_PLAYER',
            'freeThrowsMade': 'FTM_PLAYER',
            'freeThrowsAttempted': 'FTA_PLAYER',
            'freeThrowsPercentage': 'FT_PCT_PLAYER',
            'reboundsOffensive': 'OREB_PLAYER',
            'reboundsDefensive': 'DREB_PLAYER',
            'reboundsTotal': 'REB_PLAYER',
            'assists': 'AST_PLAYER',
            'turnovers': 'TOV_PLAYER',
            'steals': 'STL_PLAYER',
            'blocks': 'BLK_PLAYER',
            'plusMinusPoints': 'PLUS_MINUS_PLAYER'
        }
        
        # 重命名列
        player_df = self.player_data.copy()
        for old_col, new_col in col_mapping.items():
            if old_col in player_df.columns:
                player_df[new_col] = player_df[old_col]
        
        # 按球队聚合
        agg_cols = [col for col in col_mapping.values() if col in player_df.columns and col != 'TEAM_ABBREVIATION']
        
        if 'TEAM_ABBREVIATION' in player_df.columns and agg_cols:
            player_team_stats = player_df.groupby('TEAM_ABBREVIATION')[agg_cols].mean()
            
            # 合并到球队统计
            self.team_stats = self.team_stats.join(player_team_stats, how='left')
            print(f"✓ 已添加 {len(agg_cols)} 个球员聚合特征")
    
    def prepare_features(self):
        """准备训练特征"""
        print("\n准备训练特征...")
        
        # 泄露特征（不能用于预测）
        leakage_features = ['PLUS_MINUS', 'PLUS_MINUS_STANDARDIZED', 'PLUS_MINUS_NORMALIZED',
                           'WL', 'WL_ENCODED', 'W', 'L', 'PLUS_MINUS_REGULAR', 'PLUS_MINUS_PLAYOFF',
                           'PLUS_MINUS_PLAYER']
        
        # 选择数值特征
        feature_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                       'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                       'TOV', 'STL', 'BLK']
        feature_cols = [col for col in feature_cols if col in self.team_data.columns]
        feature_cols = [col for col in feature_cols if col not in leakage_features]
        
        self.feature_names = feature_cols
        print(f"使用特征: {feature_cols}")
        
        # 准备目标变量
        if 'WL_ENCODED' not in self.team_data.columns:
            if 'WL' in self.team_data.columns:
                self.team_data['WL_ENCODED'] = (self.team_data['WL'] == 'W').astype(int)
        
        X = self.team_data[feature_cols].fillna(self.team_data[feature_cols].mean())
        y = self.team_data['WL_ENCODED']
        
        # 删除无效行
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"训练样本数: {len(X)}")
        
        return X, y
    
    def train(self, test_size: float = 0.2, model_type: str = 'ensemble'):
        """
        训练模型
        
        Args:
            test_size: 测试集比例
            model_type: 模型类型 ('logistic', 'random_forest', 'gradient_boosting', 'ensemble')
        """
        print("\n" + "="*60)
        print("开始训练模型")
        print("="*60)
        
        X, y = self.prepare_features()
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 训练多个模型
        models = {}
        
        # 1. 逻辑回归
        print("\n训练逻辑回归模型...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        models['logistic'] = (lr_model, lr_acc)
        print(f"  逻辑回归准确率: {lr_acc:.4f}")
        
        # 2. 随机森林
        print("训练随机森林模型...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        models['random_forest'] = (rf_model, rf_acc)
        print(f"  随机森林准确率: {rf_acc:.4f}")
        
        # 3. 梯度提升
        print("训练梯度提升模型...")
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_acc = accuracy_score(y_test, gb_pred)
        models['gradient_boosting'] = (gb_model, gb_acc)
        print(f"  梯度提升准确率: {gb_acc:.4f}")
        
        # 选择最佳模型或使用集成
        if model_type == 'ensemble':
            # 集成预测
            print("\n使用集成模型...")
            self.models = {name: model for name, (model, _) in models.items()}
            self.model = None  # 集成模式
            
            # 计算集成准确率
            ensemble_pred = self._ensemble_predict_proba(X_test_scaled)
            ensemble_pred_class = (ensemble_pred > 0.5).astype(int)
            ensemble_acc = accuracy_score(y_test, ensemble_pred_class)
            print(f"  集成模型准确率: {ensemble_acc:.4f}")
            
            best_acc = ensemble_acc
        else:
            # 选择指定模型
            self.model = models[model_type][0]
            best_acc = models[model_type][1]
            self.models = None
        
        print(f"\n最终模型准确率: {best_acc:.4f}")
        
        # 特征重要性（如果是随机森林或梯度提升）
        if model_type in ['random_forest', 'gradient_boosting']:
            importance = self.model.feature_importances_
            feature_importance = sorted(zip(self.feature_names, importance), 
                                        key=lambda x: x[1], reverse=True)
            print("\n特征重要性 Top 10:")
            for feat, imp in feature_importance[:10]:
                print(f"  {feat}: {imp:.4f}")
        
        return best_acc
    
    def _ensemble_predict_proba(self, X_scaled):
        """集成模型预测概率"""
        probs = []
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[:, 1]
            probs.append(prob)
        return np.mean(probs, axis=0)
    
    def save_model(self):
        """保存模型到文件"""
        print("\n保存模型...")
        
        # 保存模型
        if self.models:
            model_path = os.path.join(self.model_dir, "nba_ensemble_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models, f)
        elif self.model:
            model_path = os.path.join(self.model_dir, "nba_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        # 保存scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存配置
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'feature_names': self.feature_names,
                'team_col': self.team_col,
                'is_ensemble': self.models is not None
            }, f, ensure_ascii=False, indent=2)
        
        # 保存球队统计数据
        stats_path = os.path.join(self.model_dir, "team_stats.csv")
        self.team_stats.to_csv(stats_path)
        
        print(f"✓ 模型已保存到 '{self.model_dir}' 文件夹")
    
    def load_model(self):
        """从文件加载模型"""
        config_path = os.path.join(self.model_dir, "config.json")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        stats_path = os.path.join(self.model_dir, "team_stats.csv")
        
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return False
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.feature_names = config['feature_names']
            self.team_col = config['team_col']
            is_ensemble = config.get('is_ensemble', False)
        
        # 加载模型
        if is_ensemble:
            model_path = os.path.join(self.model_dir, "nba_ensemble_model.pkl")
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            self.model = None
        else:
            model_path = os.path.join(self.model_dir, "nba_model.pkl")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.models = None
        
        # 加载scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 加载球队统计
        self.team_stats = pd.read_csv(stats_path, index_col=0)
        
        print("✓ 模型加载成功!")
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
        if self.model is None and self.models is None:
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
            if team1_upper in str(t).upper() or str(t).upper() in team1_upper:
                team1_match = t
            if team2_upper in str(t).upper() or str(t).upper() in team2_upper:
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
        available_features = [f for f in self.feature_names if f in self.team_stats.columns]
        stats1 = self.team_stats.loc[team1_match][available_features].values
        stats2 = self.team_stats.loc[team2_match][available_features].values
        
        # 标准化
        stats1_scaled = self.scaler.transform([stats1])
        stats2_scaled = self.scaler.transform([stats2])
        
        # 预测
        if self.models:
            # 集成预测
            prob1 = self._ensemble_predict_proba(stats1_scaled)[0]
            prob2 = self._ensemble_predict_proba(stats2_scaled)[0]
        else:
            prob1 = self.model.predict_proba(stats1_scaled)[0][1]
            prob2 = self.model.predict_proba(stats2_scaled)[0][1]
        
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
            'predicted_winner': team1_match if team1_win_prob > team2_win_prob else team2_match,
            'confidence': abs(team1_win_prob - team2_win_prob)
        }
        
        return result
    
    def predict_interactive(self):
        """交互式预测"""
        print("\n" + "="*60)
        print("NBA比赛结果预测系统 (完整版)")
        print("="*60)
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
                print("\n" + "-"*50)
                print(f"🏀 {result['team1']} vs {result['team2']}")
                print(f"   {result['team1']} 胜率: {result['team1_win_prob']:.1%}")
                print(f"   {result['team2']} 胜率: {result['team2_win_prob']:.1%}")
                print(f"   预测获胜: {result['predicted_winner']} 🏆")
                print(f"   置信度: {result['confidence']:.1%}")
                print("-"*50 + "\n")


def train_and_save():
    """训练并保存模型"""
    predictor = NBAFullModelPredictor()
    predictor.load_all_data()
    predictor.train(model_type='ensemble')
    predictor.save_model()
    return predictor


def load_and_predict():
    """加载模型并进行预测"""
    predictor = NBAFullModelPredictor()
    
    if not predictor.load_model():
        print("模型不存在，开始训练...")
        predictor = train_and_save()
    
    return predictor


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import sys
    
    # 检查是否有命令行参数
    if len(sys.argv) >= 3:
        # 命令行模式: python nba_full_model_predictor.py LAL GSW
        predictor = load_and_predict()
        result = predictor.predict(sys.argv[1], sys.argv[2])
        if result:
            print(f"\n{result['team1']} vs {result['team2']}")
            print(f"{result['team1']} 胜率: {result['team1_win_prob']:.1%}")
            print(f"{result['team2']} 胜率: {result['team2_win_prob']:.1%}")
            print(f"预测获胜: {result['predicted_winner']}")
            print(f"置信度: {result['confidence']:.1%}")
    else:
        # 交互模式
        predictor = load_and_predict()
        predictor.predict_interactive()
