"""
NBA比赛预测 - 交互式版本
"""

from nba_transformer_predictor import NBATransformerPredictor
import sys


def show_available_teams(predictor):
    """显示可用的球队"""
    teams = sorted(predictor.raw_data['TEAM_ABBREVIATION'].unique())
    print("\n📋 可用球队列表:")
    print("-" * 50)
    for i, team in enumerate(teams, 1):
        team_full = predictor.raw_data[predictor.raw_data['TEAM_ABBREVIATION'] == team]['TEAM_NAME'].iloc[0]
        print(f"{i:2d}. {team:>4} - {team_full}")
    print("-" * 50)


def main():
    print("="*70)
    print("🏀 NBA比赛预测系统 - 交互式版本")
    print("="*70)
    
    # 创建预测器
    predictor = NBATransformerPredictor(window_size=3)
    
    # 加载模型
    print("\n📦 加载模型...")
    if not predictor.load_model():
        print("❌ 模型不存在，请先训练模型: python train_transformer.py")
        return
    
    # 加载数据
    print("📊 加载数据...")
    predictor.load_data()
    predictor.prepare_features()
    
    print("\n✅ 系统准备就绪!")
    
    while True:
        print("\n" + "="*70)
        print("请输入要预测的比赛 (输入 'list' 查看球队列表, 'quit' 退出)")
        print("="*70)
        
        # 输入球队1
        team1 = input("\n🏀 球队1 (名称或缩写): ").strip()
        if team1.lower() == 'quit':
            print("\n👋 再见!")
            break
        if team1.lower() == 'list':
            show_available_teams(predictor)
            continue
        
        # 输入球队2
        team2 = input("🏀 球队2 (名称或缩写): ").strip()
        if team2.lower() == 'quit':
            print("\n👋 再见!")
            break
        if team2.lower() == 'list':
            show_available_teams(predictor)
            continue
        
        # 预测
        print("\n⏳ 预测中...")
        try:
            result = predictor.predict_by_team_names(
                team1, 
                team2, 
                save_attention=True,
                output_dir='prediction_output'
            )
            
            # 显示结果
            print("\n" + "="*70)
            print("📊 预测结果")
            print("="*70)
            
            team1_name = result['team1_name']
            team2_name = result['team2_name']
            team1_prob = result['team1_win_prob']
            team2_prob = result['team2_win_prob']
            winner = team1_name if result['predicted_winner'] == 'team1' else team2_name
            confidence = result['confidence']
            
            print(f"\n⚔️  {team1_name} vs {team2_name}")
            print("-" * 70)
            
            # 可视化概率条
            bar_length = 50
            team1_bar = int(team1_prob * bar_length)
            team2_bar = int(team2_prob * bar_length)
            
            print(f"\n{team1_name:>6} |{'█' * team1_bar}{' ' * (bar_length - team1_bar)}| {team1_prob:.1%} {'🏆' if winner == team1_name else ''}")
            print(f"{team2_name:>6} |{'█' * team2_bar}{' ' * (bar_length - team2_bar)}| {team2_prob:.1%} {'🏆' if winner == team2_name else ''}")
            
            print(f"\n✨ 预测获胜: {winner}")
            print(f"💪 置信度: {confidence:.2%}")
            
            print("\n📈 注意力热力图已保存到: prediction_output/")
            print(f"   - attention_{team1_name}.png")
            print(f"   - attention_{team2_name}.png")
            
        except Exception as e:
            print(f"\n❌ 预测失败: {e}")
            print("\n💡 提示: 输入 'list' 查看可用球队列表")
        
        # 继续预测
        continue_pred = input("\n❓ 继续预测? (y/n): ").strip().lower()
        if continue_pred != 'y':
            print("\n👋 再见!")
            break


if __name__ == "__main__":
    main()
