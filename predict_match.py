"""
NBA比赛预测脚本 - 根据队名预测比赛结果并可视化注意力
"""

from nba_transformer_predictor import NBATransformerPredictor
import argparse


def main():
    parser = argparse.ArgumentParser(description='预测NBA比赛结果')
    parser.add_argument('team1', type=str, help='球队1名称或缩写 (例如: LAL, Lakers)')
    parser.add_argument('team2', type=str, help='球队2名称或缩写 (例如: GSW, Warriors)')
    parser.add_argument('--no-attention', action='store_true', help='不保存注意力热力图')
    parser.add_argument('--output-dir', type=str, default='prediction_output', help='输出目录')
    parser.add_argument('--window-size', type=int, default=3, help='滑动窗口大小')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🏀 NBA比赛预测系统")
    print("="*70)
    
    # 创建预测器
    predictor = NBATransformerPredictor(window_size=args.window_size)
    
    # 加载模型
    print("\n📦 加载模型...")
    if not predictor.load_model():
        print("❌ 模型不存在，请先训练模型: python train_transformer.py")
        return
    
    # 加载数据
    print("\n📊 加载数据...")
    predictor.load_data()
    predictor.prepare_features()
    
    # 预测
    print("\n" + "="*70)
    print(f"⚔️  {args.team1.upper()} vs {args.team2.upper()}")
    print("="*70)
    
    try:
        result = predictor.predict_by_team_names(
            args.team1, 
            args.team2, 
            save_attention=not args.no_attention,
            output_dir=args.output_dir
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
        
        print(f"\n🏀 {team1_name:>6} 胜率: {team1_prob:6.2%}  {'🏆' if winner == team1_name else ''}")
        print(f"🏀 {team2_name:>6} 胜率: {team2_prob:6.2%}  {'🏆' if winner == team2_name else ''}")
        print(f"\n✨ 预测获胜: {winner}")
        print(f"💪 置信度: {confidence:.2%}")
        
        # 可视化概率条
        print("\n" + "-"*70)
        bar_length = 50
        team1_bar = int(team1_prob * bar_length)
        team2_bar = int(team2_prob * bar_length)
        
        print(f"{team1_name:>6} |{'█' * team1_bar}{' ' * (bar_length - team1_bar)}| {team1_prob:.1%}")
        print(f"{team2_name:>6} |{'█' * team2_bar}{' ' * (bar_length - team2_bar)}| {team2_prob:.1%}")
        print("-"*70)
        
        if not args.no_attention:
            print(f"\n📈 注意力热力图已保存到: {args.output_dir}/")
            print(f"   - attention_{team1_name}.png")
            print(f"   - attention_{team2_name}.png")
        
        print("\n" + "="*70)
        print("✅ 预测完成!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        print("\n💡 提示:")
        print("   - 确保球队名称正确 (例如: LAL, Lakers, GSW, Warriors)")
        print("   - 查看可用球队: python -c \"from nba_transformer_predictor import *; p=NBATransformerPredictor(); p.load_data(); print(sorted(p.raw_data['TEAM_ABBREVIATION'].unique()))\"")


if __name__ == "__main__":
    main()
