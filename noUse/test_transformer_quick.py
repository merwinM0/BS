"""
快速测试Transformer模型代码的正确性
不需要实际训练，只测试代码结构
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试导入"""
    print("测试1: 检查导入...")
    try:
        from nba_transformer_predictor import (
            NBATransformerPredictor,
            TransformerPredictor,
            PositionalEncoding,
            NBAGameDataset
        )
        print("✓ 所有类导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_model_structure():
    """测试模型结构"""
    print("\n测试2: 检查模型结构...")
    try:
        import torch
        from nba_transformer_predictor import TransformerPredictor
        
        # 创建模型
        model = TransformerPredictor(feature_dim=18, d_model=64, nhead=4, num_layers=2)
        
        # 测试前向传播
        batch_size = 4
        seq_len = 3
        feature_dim = 18
        x = torch.randn(batch_size, seq_len, feature_dim)
        output = model(x)
        
        assert output.shape == (batch_size, 2), f"输出形状错误: {output.shape}"
        print(f"✓ 模型结构正确，输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 模型结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n测试3: 检查数据加载...")
    try:
        from nba_transformer_predictor import NBATransformerPredictor
        
        predictor = NBATransformerPredictor(window_size=3)
        
        # 检查数据文件是否存在
        data_file = os.path.join(predictor.data_dir, 'regular_season_totals_2010_2024.csv')
        if not os.path.exists(data_file):
            print(f"✗ 数据文件不存在: {data_file}")
            return False
        
        print(f"✓ 数据文件存在: {data_file}")
        
        # 尝试加载数据
        predictor.load_data()
        print(f"✓ 数据加载成功: {len(predictor.raw_data)} 条记录")
        
        # 准备特征
        predictor.prepare_features()
        print(f"✓ 特征准备成功: {len(predictor.feature_names)} 个特征")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_creation():
    """测试序列创建"""
    print("\n测试4: 检查序列创建...")
    try:
        from nba_transformer_predictor import NBATransformerPredictor
        
        predictor = NBATransformerPredictor(window_size=3)
        predictor.load_data()
        predictor.prepare_features()
        predictor.create_sequences(test_year='2023-24')
        
        print(f"✓ 训练集序列: {predictor.train_sequences.shape}")
        print(f"✓ 测试集序列: {predictor.test_sequences.shape}")
        print(f"✓ 训练集标签: {predictor.train_labels.shape}")
        print(f"✓ 测试集标签: {predictor.test_labels.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 序列创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("NBA Transformer模型代码测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_imports),
        ("模型结构测试", test_model_structure),
        ("数据加载测试", test_data_loading),
        ("序列创建测试", test_sequence_creation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}发生异常: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！模型代码正确。")
        print("现在可以运行: python train_transformer.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
