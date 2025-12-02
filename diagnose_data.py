"""
数据诊断脚本 - 检查数据质量和潜在问题
"""

import pandas as pd
import numpy as np
import os
from collections import Counter


def diagnose_data():
    """诊断NBA数据集"""
    
    print("="*70)
    print("NBA数据诊断工具")
    print("="*70)
    
    # 加载数据
    data_dir = './NBA-Data-2010-2024-main'
    regular_file = os.path.join(data_dir, 'regular_season_totals_2010_2024.csv')
    
    if not os.path.exists(regular_file):
        print(f"❌ 错误: 找不到数据文件 {regular_file}")
        return
    
    df = pd.read_csv(regular_file)
    print(f"✓ 成功加载数据: {len(df)} 条记录")
    print(f"  列数: {len(df.columns)}")
    print(f"  时间范围: {df['SEASON_YEAR'].min()} - {df['SEASON_YEAR'].max()}")
    
    # 1. 检查胜负分布
    print("\n" + "="*70)
    print("1. 胜负分布检查")
    print("="*70)
    
    wl_counts = df['WL'].value_counts()
    total = len(df)
    print(f"总比赛数: {total}")
    for wl, count in wl_counts.items():
        pct = count / total * 100
        print(f"  {wl}: {count:5d} ({pct:.2f}%)")
    
    # 按赛季检查
    print("\n按赛季分布:")
    for season in sorted(df['SEASON_YEAR'].unique()):
        season_df = df[df['SEASON_YEAR'] == season]
        win_rate = (season_df['WL'] == 'W').mean()
        print(f"  {season}: {len(season_df):4d} 场, 胜率 {win_rate:.2%}")
    
    # 2. 检查缺失值
    print("\n" + "="*70)
    print("2. 缺失值检查")
    print("="*70)
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✓ 没有缺失值")
    else:
        print(f"发现 {len(missing)} 列有缺失值:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    
    # 3. 检查数值特征分布
    print("\n" + "="*70)
    print("3. 数值特征统计")
    print("="*70)
    
    # 排除非特征列
    exclude_cols = [
        'SEASON_YEAR', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
        'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'AVAILABLE_FLAG'
    ]
    
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"数值特征数量: {len(numeric_cols)}")
    
    # 检查是否有常数列
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\n⚠️  发现 {len(constant_cols)} 个常数列 (所有值相同):")
        for col in constant_cols:
            print(f"  - {col}")
    else:
        print("✓ 没有常数列")
    
    # 检查异常值
    print("\n异常值检查 (前5个特征示例):")
    for col in numeric_cols[:5]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            pct = outliers / len(df) * 100
            print(f"  {col}: {outliers} 个异常值 ({pct:.2f}%)")
    
    # 4. 检查潜在的数据泄露
    print("\n" + "="*70)
    print("4. 数据泄露检查")
    print("="*70)
    
    # 检查PLUS_MINUS与胜负的相关性
    if 'PLUS_MINUS' in df.columns:
        df_temp = df.copy()
        df_temp['label'] = (df_temp['WL'] == 'W').astype(int)
        correlation = df_temp['PLUS_MINUS'].corr(df_temp['label'])
        print(f"PLUS_MINUS 与胜负相关性: {correlation:.4f}")
        if abs(correlation) > 0.8:
            print("  ⚠️  高度相关! 这是数据泄露,不应使用此特征")
        else:
            print("  ✓ 相关性可接受")
    
    # 检查RANK列
    rank_cols = [col for col in df.columns if '_RANK' in col]
    if rank_cols:
        print(f"\n发现 {len(rank_cols)} 个RANK列:")
        for col in rank_cols[:5]:
            print(f"  - {col}")
        print("  ⚠️  RANK列可能包含未来信息,建议排除")
    
    # 5. 检查特征方差
    print("\n" + "="*70)
    print("5. 特征方差检查")
    print("="*70)
    
    low_variance_cols = []
    for col in numeric_cols:
        if df[col].std() < 0.01:
            low_variance_cols.append(col)
    
    if low_variance_cols:
        print(f"发现 {len(low_variance_cols)} 个低方差特征 (std < 0.01):")
        for col in low_variance_cols[:10]:
            print(f"  - {col}: std={df[col].std():.6f}")
    else:
        print("✓ 所有特征都有足够的方差")
    
    # 6. 检查类别不平衡
    print("\n" + "="*70)
    print("6. 类别平衡检查")
    print("="*70)
    
    win_rate = (df['WL'] == 'W').mean()
    imbalance_ratio = max(win_rate, 1 - win_rate) / min(win_rate, 1 - win_rate)
    
    print(f"总体胜率: {win_rate:.2%}")
    print(f"不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        print("  ⚠️  类别不平衡! 建议使用类别权重")
    else:
        print("  ✓ 类别基本平衡")
    
    # 按球队检查
    print("\n各球队胜率分布:")
    team_stats = []
    for team in df['TEAM_ABBREVIATION'].unique():
        team_df = df[df['TEAM_ABBREVIATION'] == team]
        win_rate = (team_df['WL'] == 'W').mean()
        team_stats.append((team, len(team_df), win_rate))
    
    team_stats.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'球队':<6} {'比赛数':<8} {'胜率':<8}")
    print("-" * 25)
    for team, games, wr in team_stats[:5]:
        print(f"{team:<6} {games:<8} {wr:.2%}")
    print("  ...")
    for team, games, wr in team_stats[-5:]:
        print(f"{team:<6} {games:<8} {wr:.2%}")
    
    # 7. 推荐配置
    print("\n" + "="*70)
    print("7. 推荐训练配置")
    print("="*70)
    
    # 计算推荐的类别权重
    win_count = (df['WL'] == 'W').sum()
    loss_count = (df['WL'] == 'L').sum()
    total = win_count + loss_count
    weight_loss = total / (2 * loss_count)
    weight_win = total / (2 * win_count)
    
    print(f"推荐类别权重: [Loss={weight_loss:.3f}, Win={weight_win:.3f}]")
    
    # 推荐学习率
    n_samples = len(df)
    if n_samples < 1000:
        rec_lr = 0.001
    elif n_samples < 10000:
        rec_lr = 0.0001
    else:
        rec_lr = 0.00005
    
    print(f"推荐学习率: {rec_lr}")
    print(f"推荐批次大小: {min(64, n_samples // 100)}")
    
    # 推荐窗口大小
    avg_games_per_team = df.groupby('TEAM_ABBREVIATION').size().mean()
    rec_window = min(5, int(avg_games_per_team * 0.05))
    print(f"推荐窗口大小: {rec_window}")
    
    print("\n" + "="*70)
    print("诊断完成!")
    print("="*70)
    
    # 总结
    print("\n📊 诊断总结:")
    issues = []
    
    if len(missing) > 0:
        issues.append(f"- 有 {len(missing)} 列存在缺失值")
    
    if constant_cols:
        issues.append(f"- 有 {len(constant_cols)} 个常数列需要移除")
    
    if imbalance_ratio > 1.5:
        issues.append(f"- 类别不平衡 ({imbalance_ratio:.2f}:1)")
    
    if rank_cols:
        issues.append(f"- 有 {len(rank_cols)} 个RANK列可能泄露信息")
    
    if issues:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ 数据质量良好,可以开始训练!")


if __name__ == "__main__":
    diagnose_data()
