import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
warnings.filterwarnings('ignore')

class TotalsDataProcessor:
    """
    NBA Totals数据预处理器
    松耦合设计，只需传入文件路径即可处理
    """
    
    def __init__(self, file_path: str, output_dir: str = "processed_data"):
        """
        初始化处理器
        
        Args:
            file_path: CSV文件路径
            output_dir: 输出目录，默认为processed_data
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        self.validation_results = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """加载数据"""
        try:
            self.raw_data = pd.read_csv(self.file_path)
            print(f"成功加载数据: {self.raw_data.shape[0]} 行, {self.raw_data.shape[1]} 列")
            return self.raw_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def data_quality_check(self) -> Dict:
        """
        1. 数据质量检查
        """
        print("=== 开始数据质量检查 ===")
        results = {}
        
        if self.raw_data is None:
            results['error'] = '数据未加载'
            return results
        
        # 缺失值检查
        missing_values = self.raw_data.isnull().sum()
        results['missing_values'] = missing_values[missing_values > 0].to_dict()
        print(f"缺失值统计: {results['missing_values']}")
        
        # 数据类型检查
        numeric_columns = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                           'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 
                           'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
        
        for col in numeric_columns:
            if col in self.raw_data.columns:
                # 转换为数值类型
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
        
        results['data_types'] = self.raw_data.dtypes.to_dict()
        print("数据类型转换完成")
        
        # 异常值检测
        results['outliers'] = self._detect_outliers()
        print(f"异常值检测完成: 发现 {len(results['outliers'])} 个异常值")
        
        return results
    
    def _detect_outliers(self) -> List[Dict]:
        """检测异常值"""
        outliers = []
        
        if self.raw_data is None:
            return outliers
        
        numeric_columns = ['PTS', 'FGM', 'REB', 'AST', 'STL', 'BLK']
        
        for col in numeric_columns:
            if col in self.raw_data.columns:
                Q1 = self.raw_data[col].quantile(0.25)
                Q3 = self.raw_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.raw_data[col] < lower_bound) | (self.raw_data[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers.append({
                        'column': col,
                        'count': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
        
        return outliers
    
    def data_cleaning(self) -> Dict:
        """
        2. 数据清洗
        """
        print("=== 开始数据清洗 ===")
        results = {}
        
        if self.raw_data is None:
            results['error'] = '数据未加载'
            return results
        
        # 重复数据检查
        initial_count = len(self.raw_data)
        duplicates = self.raw_data.duplicated().sum()
        self.raw_data = self.raw_data.drop_duplicates()
        results['duplicates_removed'] = duplicates
        print(f"移除重复数据: {duplicates} 行")
        
        # 格式标准化
        self.raw_data['GAME_DATE'] = pd.to_datetime(self.raw_data['GAME_DATE'])
        self.raw_data['SEASON_YEAR'] = self.raw_data['SEASON_YEAR'].astype(str)
        results['format_standardization'] = "日期和赛季格式标准化完成"
        
        # 无效数据过滤 (totals文件通常没有DNP记录，但检查一下)
        if 'WL' in self.raw_data.columns:
            invalid_count = self.raw_data['WL'].isnull().sum()
            self.raw_data = self.raw_data.dropna(subset=['WL'])
            results['invalid_data_removed'] = invalid_count
            print(f"移除无效数据: {invalid_count} 行")
        
        return results
    
    def feature_engineering(self) -> Dict:
        """
        3. 特征工程
        """
        print("=== 开始特征工程 ===")
        results = {}
        
        if self.raw_data is None:
            results['error'] = '数据未加载'
            return results
        
        # 计算衍生指标
        self._calculate_derived_metrics()
        results['derived_metrics'] = ['FG_EFFICIENCY', 'TRUE_SHOOTING_PCT', 'OFFENSIVE_RATING']
        
        # 时间特征提取
        self._extract_time_features()
        results['time_features'] = ['MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND']
        
        # 对手实力评估
        self._calculate_opponent_strength()
        results['opponent_strength'] = 'OPPONENT_STRENGTH_SCORE'
        
        return results
    
    def _calculate_derived_metrics(self):
        """计算衍生指标"""
        if self.raw_data is None:
            return
        
        # 投篮效率 (FG_EFFICIENCY = PTS / FGA)
        self.raw_data['FG_EFFICIENCY'] = np.where(
            self.raw_data['FGA'] > 0,
            self.raw_data['PTS'] / self.raw_data['FGA'],
            0
        )
        
        # 真实投篮命中率 (TS_PCT = PTS / (2 * (FGA + 0.44 * FTA)))
        self.raw_data['TRUE_SHOOTING_PCT'] = np.where(
            (self.raw_data['FGA'] + 0.44 * self.raw_data['FTA']) > 0,
            self.raw_data['PTS'] / (2 * (self.raw_data['FGA'] + 0.44 * self.raw_data['FTA'])),
            0
        )
        
        # 进攻效率评分 (综合得分、助攻、篮板)
        self.raw_data['OFFENSIVE_RATING'] = (
            self.raw_data['PTS'] * 0.4 + 
            self.raw_data['AST'] * 0.3 + 
            self.raw_data['REB'] * 0.3
        )
    
    def _extract_time_features(self):
        """提取时间特征"""
        if self.raw_data is None:
            return
        
        self.raw_data['MONTH'] = self.raw_data['GAME_DATE'].dt.month
        self.raw_data['DAY_OF_WEEK'] = self.raw_data['GAME_DATE'].dt.dayofweek
        self.raw_data['IS_WEEKEND'] = (self.raw_data['DAY_OF_WEEK'] >= 5).astype(int)
    
    def _calculate_opponent_strength(self):
        """计算对手实力评估"""
        if self.raw_data is None:
            return
        
        # 预先计算每个球队的平均得分
        team_avg_pts = self.raw_data.groupby('TEAM_ABBREVIATION')['PTS'].mean().to_dict()
        
        # 使用向量化操作提取对手信息
        def extract_opponent(matchup):
            if '@' in matchup:
                # 客场比赛
                return matchup.split(' @ ')[1].split()[0]
            elif 'vs.' in matchup:
                # 主场比赛
                return matchup.split(' vs. ')[1].split()[0]
            else:
                return None  # 默认值
        
        # 提取对手缩写
        opponents = self.raw_data['MATCHUP'].apply(extract_opponent)
        
        # 映射对手实力评分
        self.raw_data['OPPONENT_STRENGTH_SCORE'] = (
            opponents.map(team_avg_pts).fillna(50) / 100  # 默认50分，归一化
        ).fillna(0.5)  # 如果还是没有，使用0.5
    
    def data_integration(self) -> Dict:
        """
        4. 数据整合 (totals文件单独处理)
        """
        print("=== 开始数据整合 ===")
        results = {}
        
        if self.raw_data is None:
            results['error'] = '数据未加载'
            return results
        
        # 球队统计汇总
        team_stats = self._calculate_team_aggregates()
        results['team_aggregates'] = list(team_stats.keys())
        
        # 赛季统计计算
        season_stats = self._calculate_season_stats()
        results['season_stats'] = list(season_stats.keys())
        
        return results
    
    def _calculate_team_aggregates(self) -> Dict:
        """计算球队汇总统计"""
        team_aggregates = {}
        
        if self.raw_data is None:
            return team_aggregates
        
        # 按球队分组计算各种统计
        groupby_team = self.raw_data.groupby('TEAM_ID')
        
        team_aggregates['TEAM_AVG_PTS'] = groupby_team['PTS'].mean()
        team_aggregates['TEAM_TOTAL_GAMES'] = groupby_team.size()
        team_aggregates['TEAM_WIN_RATE'] = (self.raw_data[self.raw_data['WL'] == 'W']
                                            .groupby('TEAM_ID').size() / 
                                            groupby_team.size()).fillna(0)
        
        return team_aggregates
    
    def _calculate_season_stats(self) -> Dict:
        """计算赛季统计"""
        season_stats = {}
        
        if self.raw_data is None:
            return season_stats
        
        groupby_season = self.raw_data.groupby('SEASON_YEAR')
        
        season_stats['SEASON_AVG_PTS'] = groupby_season['PTS'].mean()
        season_stats['SEASON_TOTAL_GAMES'] = groupby_season.size()
        season_stats['SEASON_TEAMS_COUNT'] = groupby_season['TEAM_ID'].nunique()
        
        return season_stats
    
    def data_standardization(self) -> Dict:
        """
        5. 数据标准化
        """
        print("=== 开始数据标准化 ===")
        results = {}
        
        if self.raw_data is None:
            results['error'] = '数据未加载'
            return results
        
        # 数值标准化 (z-score) - 使用StandardScaler
        numeric_columns = ['PTS', 'FGM', 'REB', 'AST', 'STL', 'BLK', 'PLUS_MINUS']
        available_columns = [col for col in numeric_columns if col in self.raw_data.columns]
        
        if available_columns:
            # 标准化
            standard_scaler = StandardScaler()
            standardized_data = standard_scaler.fit_transform(self.raw_data[available_columns])
            
            for i, col in enumerate(available_columns):
                self.raw_data[f'{col}_STANDARDIZED'] = standardized_data[:, i]
            
            results['standardization'] = {
                'scaler_type': 'StandardScaler',
                'columns': available_columns,
                'mean': standard_scaler.mean_.tolist(),
                'scale': standard_scaler.scale_.tolist()
            }
            
            # 归一化处理 (0-1范围) - 使用MinMaxScaler
            minmax_scaler = MinMaxScaler()
            normalized_data = minmax_scaler.fit_transform(self.raw_data[available_columns])
            
            for i, col in enumerate(available_columns):
                self.raw_data[f'{col}_NORMALIZED'] = normalized_data[:, i]
            
            results['normalization'] = {
                'scaler_type': 'MinMaxScaler',
                'columns': available_columns,
                'data_min': minmax_scaler.data_min_.tolist(),
                'data_max': minmax_scaler.data_max_.tolist()
            }
        
        # 分类编码 - 使用LabelEncoder
        self._encode_categorical_variables()
        results['categorical_encoding'] = ['TEAM_ID_ENCODED', 'WL_ENCODED']
        
        return results
    
    def _encode_categorical_variables(self):
        """编码分类变量 - 使用LabelEncoder"""
        if self.raw_data is None:
            return
        
        # 球队ID编码 - 使用LabelEncoder
        team_encoder = LabelEncoder()
        self.raw_data['TEAM_ID_ENCODED'] = team_encoder.fit_transform(self.raw_data['TEAM_ID'])
        
        # 比赛结果编码 - 使用LabelEncoder
        wl_encoder = LabelEncoder()
        # 处理可能的NaN值
        wl_data = self.raw_data['WL'].fillna('UNKNOWN')
        self.raw_data['WL_ENCODED'] = wl_encoder.fit_transform(wl_data)
        # 将UNKNOWN映射回-1
        self.raw_data.loc[self.raw_data['WL'].isna(), 'WL_ENCODED'] = -1
    
    def data_validation(self) -> Dict:
        """
        6. 数据验证
        """
        print("=== 开始数据验证 ===")
        results = {}
        
        if self.raw_data is None:
            results['error'] = '数据未加载'
            return results
        
        # 逻辑检查
        logic_errors = self._logic_validation()
        results['logic_validation'] = logic_errors
        
        # 范围检查
        range_errors = self._range_validation()
        results['range_validation'] = range_errors
        
        # 完整性验证
        completeness_errors = self._completeness_validation()
        results['completeness_validation'] = completeness_errors
        
        return results
    
    def _logic_validation(self) -> List[Dict]:
        """逻辑检查"""
        errors = []
        
        if self.raw_data is None:
            return errors
        
        # 检查得分与投篮数的一致性
        # PTS应该约等于 FGM*2 + FG3M*1 + FTM*1
        expected_pts = self.raw_data['FGM'] * 2 + self.raw_data['FG3M'] + self.raw_data['FTM']
        pts_diff = abs(self.raw_data['PTS'] - expected_pts)
        
        inconsistent_mask = pts_diff > 5  # 允许5分误差
        if inconsistent_mask.any():
            errors.append({
                'type': 'score_consistency',
                'count': inconsistent_mask.sum(),
                'description': '得分与投篮数不一致'
            })
        
        return errors
    
    def _range_validation(self) -> List[Dict]:
        """范围检查"""
        errors = []
        
        if self.raw_data is None:
            return errors
        
        # 检查负数统计
        numeric_columns = ['PTS', 'FGM', 'FGA', 'REB', 'AST', 'STL', 'BLK']
        for col in numeric_columns:
            if col in self.raw_data.columns:
                negative_count = (self.raw_data[col] < 0).sum()
                if negative_count > 0:
                    errors.append({
                        'type': 'negative_values',
                        'column': col,
                        'count': negative_count,
                        'description': f'{col}存在负数值'
                    })
        
        # 检查命中率范围
        pct_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        for col in pct_columns:
            if col in self.raw_data.columns:
                out_of_range = ((self.raw_data[col] < 0) | (self.raw_data[col] > 1)).sum()
                if out_of_range > 0:
                    errors.append({
                        'type': 'percentage_range',
                        'column': col,
                        'count': out_of_range,
                        'description': f'{col}超出0-1范围'
                    })
        
        return errors
    
    def _completeness_validation(self) -> List[Dict]:
        """完整性验证"""
        errors = []
        
        if self.raw_data is None:
            return errors
        
        # 检查关键字段的完整性
        required_fields = ['GAME_ID', 'TEAM_ID', 'SEASON_YEAR', 'WL']
        for field in required_fields:
            if field in self.raw_data.columns:
                missing_count = self.raw_data[field].isnull().sum()
                if missing_count > 0:
                    errors.append({
                        'type': 'missing_required_field',
                        'field': field,
                        'count': missing_count,
                        'description': f'关键字段{field}缺失'
                    })
        
        return errors
    
    def process(self, save_output: bool = True) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        执行完整的数据预处理流程
        
        Args:
            save_output: 是否自动保存处理后的数据，默认为True
        
        Returns:
            Tuple[Optional[pd.DataFrame], Dict]: 处理后的数据和所有处理结果
        """
        print("开始NBA Totals数据预处理...")
        
        # 加载数据
        if self.load_data() is None:
            return None, {}
        
        # 执行所有处理步骤
        all_results = {}
        
        all_results['data_quality'] = self.data_quality_check()
        all_results['data_cleaning'] = self.data_cleaning()
        all_results['feature_engineering'] = self.feature_engineering()
        all_results['data_integration'] = self.data_integration()
        all_results['data_standardization'] = self.data_standardization()
        all_results['data_validation'] = self.data_validation()
        
        self.processed_data = self.raw_data.copy() if self.raw_data is not None else None
        self.validation_results = all_results
        
        print("=== 数据预处理完成 ===")
        if self.processed_data is not None:
            print(f"处理后数据形状: {self.processed_data.shape}")
            duplicates_removed = all_results['data_cleaning'].get('duplicates_removed', 0)
            duplicates_count = int(duplicates_removed) if hasattr(duplicates_removed, '__len__') else duplicates_removed
            print(f"新增列数: {self.processed_data.shape[1] - self.raw_data.shape[1] + duplicates_count}")
            
            # 自动保存处理后的数据
            if save_output:
                self.save_processed_data()
        else:
            print("数据处理失败")
        
        return self.processed_data, all_results
    
    def save_processed_data(self) -> bool:
        """
        保存处理后的数据到processed_data文件夹，只保存新增的列
        
        Returns:
            bool: 保存是否成功
        """
        if self.processed_data is None:
            print("没有处理后的数据可保存")
            return False
        
        try:
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_processed.csv")
            
            # 获取原始列名（假设原始数据已加载）
            original_columns = set()
            if self.raw_data is not None:
                # 重新加载原始数据以获取原始列名
                try:
                    original_data = pd.read_csv(self.file_path)
                    original_columns = set(original_data.columns)
                except:
                    # 如果无法重新加载，使用常见的原始列名
                    original_columns = {
                        'GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_DATE',
                        'MATCHUP', 'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                        'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
                        'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                        'SEASON_YEAR'
                    }
            
            # 获取新增列
            all_columns = set(self.processed_data.columns)
            new_columns = list(all_columns - original_columns)
            
            if new_columns:
                # 只保存新增列
                new_data = self.processed_data[new_columns]
                new_data.to_csv(output_file, index=False)
                print(f"新增列数据已保存到: {output_file}")
                print(f"保存的新增列: {new_columns}")
            else:
                print("没有新增列需要保存")
                return True
            
            # 保存处理结果摘要
            summary_file = os.path.join(self.output_dir, f"{base_name}_summary.json")
            import json
            summary_data = self.validation_results.copy()
            summary_data['new_columns'] = new_columns
            summary_data['original_columns_count'] = len(original_columns)
            summary_data['new_columns_count'] = len(new_columns)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"处理结果摘要已保存到: {summary_file}")
            
            return True
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False
    
    def get_processed_data_info(self) -> Dict:
        """获取处理后数据的信息"""
        if self.processed_data is None:
            return {}
        
        validation_data = self.validation_results.get('data_validation', {})
        total_errors = sum(len(v) if isinstance(v, list) else 1 
                          for v in validation_data.values() if not isinstance(v, dict) or 'error' not in v)
        
        return {
            'shape': self.processed_data.shape,
            'columns': list(self.processed_data.columns),
            'data_types': self.processed_data.dtypes.to_dict(),
            'memory_usage': self.processed_data.memory_usage(deep=True).sum(),
            'validation_summary': {
                'total_errors': total_errors,
                'processing_steps': list(self.validation_results.keys())
            }
        }


# 使用示例
if __name__ == "__main__":
    # 处理常规赛数据
    regular_processor = TotalsDataProcessor('./NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv')
    regular_data, regular_results = regular_processor.process()
    
    print("\n=== 常规赛数据处理结果 ===")
    print(f"数据形状: {regular_data.shape if regular_data is not None else 'None'}")
    print(f"处理步骤: {list(regular_results.keys())}")
    
    # 处理季后赛数据
    playoff_processor = TotalsDataProcessor('./NBA-Data-2010-2024-main/play_off_totals_2010_2024.csv')
    playoff_data, playoff_results = playoff_processor.process()
    
    print("\n=== 季后赛数据处理结果 ===")
    print(f"数据形状: {playoff_data.shape if playoff_data is not None else 'None'}")
    print(f"处理步骤: {list(playoff_results.keys())}")
    
    # 显示保存的文件信息
    print("\n=== 文件保存信息 ===")
    print("处理后的数据已保存到 processed_data 文件夹:")
    print("- regular_season_totals_2010_2024_processed.csv")
    print("- regular_season_totals_2010_2024_summary.json")
    print("- play_off_totals_2010_2024_processed.csv")
    print("- play_off_totals_2010_2024_summary.json")