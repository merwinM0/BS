from nba_transformer_predictor import NBATransformerPredictor
import argparse


def main():
    parser = argparse.ArgumentParser(description='训练NBA Transformer预测模型')
    parser.add_argument('--window_size', type=int, default=10, help='滑动窗口大小(默认10场比赛)')
    parser.add_argument('--test_year', type=str, default='2023-24', help='测试集年份(默认2023-24)')
    parser.add_argument('--d_model', type=int, default=1024, help='Transformer模型维度(默认1024)')
    parser.add_argument('--nhead', type=int, default=16, help='注意力头数(默认16)')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer层数(默认4)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率(默认0.1)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数(默认100)')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小(默认256)')
    parser.add_argument('--lr', type=float, default=0.00001, help='学习率(默认0.00001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减(默认1e-4)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑(默认0.1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NBA Transformer预测模型训练")
    print("="*60)
    print("配置:")
    print(f"  滑动窗口大小: {args.window_size}")
    print(f"  测试集年份: {args.test_year}")
    print(f"  模型维度: {args.d_model}")
    print(f"  注意力头数: {args.nhead}")
    print(f"  Transformer层数: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  标签平滑: {args.label_smoothing}")
    print("="*60)
    
    # 创建预测器
    predictor = NBATransformerPredictor(window_size=args.window_size)
    
    # 加载和准备数据
    predictor.load_data()
    predictor.prepare_features()
    predictor.create_sequences(test_year=args.test_year)
    
    # 构建模型
    predictor.build_model(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # 训练模型
    best_acc = predictor.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing
    )
    
    # 评估模型
    results = predictor.evaluate()
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳测试准确率: {best_acc:.4f}")
    print(f"最终测试准确率: {results['accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
