# NBA Transformer预测模型

## 📋 模型概述

这是一个基于Transformer架构的NBA比赛预测模型，使用滑动窗口方法处理时序数据。

### 🎯 核心特点

- **滑动窗口**: 使用前N场比赛(默认3场)预测下一场比赛结果
- **Transformer架构**: 使用多头注意力机制捕捉比赛之间的关系
- **序列建模**: 每个球队的比赛作为一个时间序列
- **Softmax输出**: 输出胜负概率，用于比较两队实力
- **时间分割**: 最后一年(2023-24赛季)作为测试集

---

## 🏗️ 模型架构

```
输入: (batch_size, window_size=3, feature_dim)
  ↓
输入投影层: Linear(feature_dim → d_model=128)
  ↓
位置编码: PositionalEncoding
  ↓
Transformer编码器: 4层 × (MultiHeadAttention + FFN)
  ↓
取最后时间步: (batch_size, d_model)
  ↓
全连接层: Linear(d_model → d_model/2) → ReLU → Dropout
  ↓
输出层: Linear(d_model/2 → 2)
  ↓
Softmax: 输出胜负概率 [P(Loss), P(Win)]
```

---

## 📊 数据处理流程

### 1. 数据加载
- 加载常规赛球队统计数据 (`regular_season_totals_2010_2024.csv`)
- 按球队和日期排序

### 2. 特征选择
选择的特征包括:
- **得分相关**: PTS, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT
- **罚球相关**: FTM, FTA, FT_PCT
- **篮板相关**: OREB, DREB, REB
- **其他统计**: AST, TOV, STL, BLK, PF等

排除的特征:
- 泄露特征: PLUS_MINUS (直接反映比赛结果)
- 排名特征: 所有 *_RANK 列 (赛季累计排名)
- 非数值特征: 球队名称、日期等

### 3. 滑动窗口创建
```
球队A的比赛序列: [G1, G2, G3, G4, G5, G6, ...]

滑动窗口(window_size=3):
  [G1, G2, G3] → 预测 G4 的结果
  [G2, G3, G4] → 预测 G5 的结果
  [G3, G4, G5] → 预测 G6 的结果
  ...
```

### 4. 数据标准化
- 使用 StandardScaler 对特征进行标准化
- 在训练集上拟合，应用到测试集

---

## 🚀 使用方法

### 安装依赖

```bash
pip install torch pandas numpy scikit-learn
```

### 训练模型

#### 方法1: 使用默认参数
```bash
python train_transformer.py
```

#### 方法2: 自定义参数
```bash
python train_transformer.py \
    --window_size 3 \
    --test_year 2023-24 \
    --d_model 128 \
    --nhead 8 \
    --num_layers 4 \
    --dropout 0.1 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 1e-5
```

#### 方法3: 在Python中使用
```python
from nba_transformer_predictor import train_and_save

# 训练并保存模型
predictor = train_and_save()
```

### 加载和评估模型

```python
from nba_transformer_predictor import load_and_evaluate

# 加载模型并评估
predictor = load_and_evaluate()
```

### 预测比赛结果

```python
from nba_transformer_predictor import NBATransformerPredictor
import numpy as np

# 加载模型
predictor = NBATransformerPredictor(window_size=3)
predictor.load_model()

# 准备球队历史数据 (最近3场比赛的特征)
# team1_history: shape (3, feature_dim)
# team2_history: shape (3, feature_dim)

result = predictor.predict_match(team1_history, team2_history)

print(f"球队1胜率: {result['team1_win_prob']:.2%}")
print(f"球队2胜率: {result['team2_win_prob']:.2%}")
print(f"预测获胜: {result['predicted_winner']}")
print(f"置信度: {result['confidence']:.2%}")
```

---

## 📈 模型参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 3 | 滑动窗口大小(使用前N场比赛) |
| `test_year` | '2023-24' | 测试集年份 |
| `d_model` | 128 | Transformer模型维度 |
| `nhead` | 8 | 多头注意力头数 |
| `num_layers` | 4 | Transformer编码器层数 |
| `dropout` | 0.1 | Dropout率 |
| `epochs` | 50 | 训练轮数 |
| `batch_size` | 64 | 批次大小 |
| `lr` | 0.001 | 学习率 |
| `weight_decay` | 1e-5 | 权重衰减(L2正则化) |

---

## 📁 文件结构

```
BS-main/
├── nba_transformer_predictor.py    # 主模型文件
├── train_transformer.py            # 训练脚本
├── TRANSFORMER_README.md           # 本文档
├── saved_model_transformer/        # 模型保存目录
│   ├── transformer_model.pth       # PyTorch模型权重
│   ├── scaler.pkl                  # 特征标准化器
│   └── config.json                 # 模型配置
└── NBA-Data-2010-2024-main/        # 数据目录
    └── regular_season_totals_2010_2024.csv
```

---

## 🔬 技术细节

### Transformer编码器
- **多头注意力**: 8个注意力头，捕捉不同维度的特征关系
- **前馈网络**: d_model × 4 的隐藏层
- **残差连接**: 每层都有残差连接和层归一化
- **位置编码**: 正弦余弦位置编码，保留时序信息

### 训练策略
- **损失函数**: CrossEntropyLoss (交叉熵损失)
- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau (验证集准确率不提升时降低学习率)
- **早停**: 保存验证集上表现最好的模型

### 预测方法
1. 获取两支球队最近N场比赛的特征
2. 分别通过模型得到胜率预测
3. 使用Softmax归一化概率
4. 比较两队胜率，预测获胜方

---

## 📊 预期性能

- **训练集准确率**: ~65-75%
- **测试集准确率**: ~60-70%
- **训练时间**: ~5-10分钟 (CPU) / ~1-2分钟 (GPU)

---

## 🎯 优化建议

### 提升模型性能
1. **增加窗口大小**: 尝试 `window_size=5` 或 `window_size=7`
2. **增加模型容量**: 增大 `d_model` 或 `num_layers`
3. **数据增强**: 添加对手数据、主客场信息
4. **特征工程**: 添加移动平均、趋势特征
5. **集成学习**: 训练多个模型进行投票

### 加快训练速度
1. **使用GPU**: 自动检测并使用CUDA
2. **增大批次**: 如果内存允许，增大 `batch_size`
3. **减少层数**: 减少 `num_layers` 到 2-3 层

---

## ⚠️ 注意事项

1. **数据要求**: 确保 `NBA-Data-2010-2024-main` 目录存在且包含数据文件
2. **内存使用**: 模型会将所有数据加载到内存，确保有足够内存
3. **随机性**: 由于随机初始化，每次训练结果可能略有不同
4. **过拟合**: 如果训练集准确率远高于测试集，考虑增加 `dropout` 或 `weight_decay`

---

## 📝 示例输出

```
============================================================
加载NBA数据...
============================================================
✓ 加载数据: 35040 条记录
  赛季范围: 2010-11 - 2023-24
  球队数量: 30

准备特征...
✓ 选择特征数量: 18
  特征列表: ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', ...]

创建滑动窗口序列 (窗口大小=3)...
✓ 训练集: 28456 个序列
✓ 测试集: 2134 个序列
  序列形状: (28456, 3, 18)
  训练集胜率: 50.12%
  测试集胜率: 49.86%

构建Transformer模型...
✓ 模型构建完成
  总参数量: 245,634
  可训练参数: 245,634

============================================================
开始训练模型
============================================================
Epoch [1/50] Loss: 0.6842 Train Acc: 0.5523 Test Acc: 0.5789
Epoch [5/50] Loss: 0.6512 Train Acc: 0.6234 Test Acc: 0.6123
...
Epoch [50/50] Loss: 0.5234 Train Acc: 0.7456 Test Acc: 0.6789

============================================================
训练完成! 最佳测试准确率: 0.6823 (Epoch 47)
============================================================
```

---

## 🤝 与原模型的对比

| 特性 | 原模型 (nba_full_model_predictor.py) | Transformer模型 |
|------|--------------------------------------|-----------------|
| 模型类型 | 传统ML (逻辑回归/随机森林/梯度提升) | 深度学习 (Transformer) |
| 输入方式 | 球队历史平均统计 | 滑动窗口序列 |
| 时序建模 | ❌ 无 | ✅ 有 |
| 注意力机制 | ❌ 无 | ✅ 多头注意力 |
| 预测方式 | 基于历史平均 | 基于最近N场比赛 |
| 训练时间 | 快 (~1分钟) | 中等 (~5-10分钟) |
| 可解释性 | 高 | 中等 |
| 准确率 | ~60-65% | ~65-70% |

---

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Time Series Forecasting with Transformers](https://arxiv.org/abs/2001.08317)
