# NBA 比赛结果预测系统 - Transformer版本

> 基于深度学习Transformer架构的NBA比赛胜负预测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 项目概述

本项目使用**Transformer深度学习架构**和**滑动窗口时序建模**方法，对2010-2024年NBA比赛数据进行分析和预测。项目已经过优化，解决了训练loss不下降和准确率低的问题。

### 🎯 核心特点

- ✅ **Transformer架构**: 多头注意力机制捕捉比赛序列关系
- ✅ **滑动窗口**: 使用前N场比赛预测下一场(默认3场)
- ✅ **优化训练**: 类别权重平衡、标签平滑、梯度裁剪
- ✅ **早停机制**: 自动防止过拟合
- ✅ **完整工具链**: 数据诊断、训练、评估一体化

### 📊 模型性能

- **测试准确率**: 55-60% (专业水平)
- **训练时间**: 5-10分钟 (CPU) / 1-2分钟 (GPU)
- **模型参数**: ~245,000
- **数据规模**: 2010-2024赛季 (~30,000场比赛)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用pip安装
pip install torch pandas numpy scikit-learn

# 或使用uv包管理器
uv sync
```

### 2. 数据诊断 (推荐)

运行诊断脚本检查数据质量:

```bash
python diagnose_data.py
```

这会显示:
- 胜负分布和类别平衡
- 缺失值检查
- 数据泄露检测
- 推荐的训练参数

### 3. 训练模型

#### 使用优化后的默认参数 (推荐)

```bash
python train_transformer.py
```

#### 自定义参数训练

```bash
# 使用5场比赛窗口，训练100轮
python train_transformer.py --window_size 5 --epochs 100

# 降低学习率，增加正则化
python train_transformer.py --lr 0.00005 --weight_decay 1e-3 --dropout 0.2

# 完整参数示例
python train_transformer.py \
    --window_size 5 \
    --test_year 2023-24 \
    --d_model 256 \
    --nhead 8 \
    --num_layers 6 \
    --dropout 0.2 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.00005 \
    --weight_decay 1e-4 \
    --label_smoothing 0.15
```

### 4. 查看结果

训练完成后:
- 模型保存在 `saved_model_transformer/` 目录
- 查看每轮的详细训练日志
- 最佳模型会自动保存

---

## 🏗️ 项目结构

```
BS/
├── NBA-Data-2010-2024-main/              # 数据目录
│   └── regular_season_totals_2010_2024.csv
│
├── nba_transformer_predictor.py          # 核心模型 (主文件)
├── train_transformer.py                  # 训练脚本
├── diagnose_data.py                      # 数据诊断工具
│
├── README_CONSOLIDATED.md                # 本文档
├── TRAINING_TIPS.md                      # 训练优化指南
├── NBA数据字段说明.md                     # 数据字段文档
│
├── saved_model_transformer/              # 模型保存目录
├── old_predictor/                        # 旧版本代码
└── pyproject.toml                        # 依赖配置
```

---

## 📚 核心文件说明

### `nba_transformer_predictor.py`
主模型文件，包含:
- `TransformerPredictor`: Transformer神经网络
- `NBATransformerPredictor`: 预测器类
- `PositionalEncoding`: 位置编码
- `NBAGameDataset`: 数据集类

### `train_transformer.py`
训练脚本，支持命令行参数配置

### `diagnose_data.py`
数据诊断工具，检查:
- 数据质量和分布
- 类别平衡
- 数据泄露
- 推荐训练参数

### `TRAINING_TIPS.md`
详细的训练优化指南，包括:
- 已实施的9项改进
- 参数调优建议
- 故障排除
- 预期结果

---

## 🎯 模型架构

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
全连接层: Linear → BatchNorm → ReLU → Dropout
  ↓
输出层: Linear(d_model/2 → 2)
  ↓
输出: (batch_size, 2) - [Loss概率, Win概率]
```

### 关键技术

1. **滑动窗口**: 每个球队的历史比赛作为时间序列
2. **多头注意力**: 捕捉不同比赛之间的关系
3. **位置编码**: 保留时间顺序信息
4. **批归一化**: 加速训练，提高稳定性
5. **标签平滑**: 防止过拟合

---

## 🔧 训练优化 (已实施)

### 9项关键改进

1. **降低学习率** (0.001 → 0.0001)
   - 防止loss震荡，更稳定收敛

2. **梯度裁剪** (max_norm=1.0)
   - 防止梯度爆炸

3. **类别权重平衡**
   - 自动计算并应用权重处理不平衡

4. **标签平滑** (0.1)
   - 提高泛化能力，防止过拟合

5. **余弦退火学习率**
   - CosineAnnealingWarmRestarts调度

6. **批归一化**
   - 在全连接层添加BatchNorm

7. **Xavier初始化**
   - 更好的权重初始化

8. **早停机制** (patience=15)
   - 15轮无改善自动停止

9. **详细日志**
   - 每轮显示完整训练指标

---

## 📊 参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window_size` | 3 | 滑动窗口大小(场比赛) |
| `--d_model` | 128 | Transformer模型维度 |
| `--nhead` | 8 | 注意力头数 |
| `--num_layers` | 4 | Transformer层数 |
| `--dropout` | 0.1 | Dropout率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 0.0001 | 学习率 (已优化) |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--label_smoothing` | 0.1 | 标签平滑 |

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_year` | 2023-24 | 测试集年份 |

---

## 💡 使用示例

### 1. 预测比赛结果 (命令行)

```bash
# 预测湖人vs勇士
python predict_match.py LAL GSW

# 预测并保存注意力热力图
python predict_match.py Lakers Warriors

# 不保存注意力图
python predict_match.py LAL GSW --no-attention
```

### 2. 交互式预测

```bash
python predict_interactive.py
```

然后按提示输入球队名称，支持：
- 输入 `list` 查看所有可用球队
- 输入 `quit` 退出程序

### 3. Python代码中使用

#### 训练模型

```python
from nba_transformer_predictor import NBATransformerPredictor

# 创建预测器
predictor = NBATransformerPredictor(window_size=3)

# 加载和准备数据
predictor.load_data()
predictor.prepare_features()
predictor.create_sequences(test_year='2023-24')

# 构建模型
predictor.build_model(
    d_model=128,
    nhead=8,
    num_layers=4,
    dropout=0.1
)

# 训练
best_acc = predictor.train(
    epochs=50,
    batch_size=64,
    lr=0.0001,
    weight_decay=1e-4,
    label_smoothing=0.1
)

# 评估
results = predictor.evaluate()
print(f"测试准确率: {results['accuracy']:.2%}")
```

#### 预测比赛

```python
from nba_transformer_predictor import NBATransformerPredictor

# 加载已训练的模型
predictor = NBATransformerPredictor(window_size=3)
predictor.load_model()
predictor.load_data()
predictor.prepare_features()

# 预测湖人vs勇士，并保存注意力热力图
result = predictor.predict_by_team_names('LAL', 'GSW', save_attention=True)

print(f"{result['team1_name']} 胜率: {result['team1_win_prob']:.2%}")
print(f"{result['team2_name']} 胜率: {result['team2_win_prob']:.2%}")
print(f"预测获胜: {result['team1_name'] if result['predicted_winner']=='team1' else result['team2_name']}")
```

---

## 🔍 训练监控

### 健康的训练应该显示:

```
Epoch [  1/50] TrainLoss: 0.6931 TestLoss: 0.6928 | TrainAcc: 0.5123 TestAcc: 0.5089 | Best: 0.5089 (E1) | LR: 0.000100
Epoch [  2/50] TrainLoss: 0.6895 TestLoss: 0.6891 | TrainAcc: 0.5234 TestAcc: 0.5201 | Best: 0.5201 (E2) | LR: 0.000099
Epoch [  3/50] TrainLoss: 0.6845 TestLoss: 0.6842 | TrainAcc: 0.5456 TestAcc: 0.5423 | Best: 0.5423 (E3) | LR: 0.000098
...
Epoch [ 45/50] TrainLoss: 0.6234 TestLoss: 0.6445 | TrainAcc: 0.6234 TestAcc: 0.5789 | Best: 0.5823 (E42) | LR: 0.000023
```

### 指标说明:
- **TrainLoss/TestLoss**: 应该持续下降
- **TrainAcc/TestAcc**: 应该持续上升
- **Best**: 最佳测试准确率
- **LR**: 学习率会自动衰减

---

## ⚠️ 常见问题

### Q: Loss不下降，准确率只有50%？
**A**: 已修复! 新版本包含:
- 降低学习率到0.0001
- 添加梯度裁剪
- 类别权重平衡
- 标签平滑

### Q: 训练太慢？
**A**: 尝试:
```bash
python train_transformer.py --d_model 64 --num_layers 2 --batch_size 128
```

### Q: 过拟合 (训练准确率高，测试准确率低)？
**A**: 增加正则化:
```bash
python train_transformer.py --dropout 0.3 --weight_decay 1e-3 --label_smoothing 0.2
```

### Q: 内存不足？
**A**: 减小批次大小:
```bash
python train_transformer.py --batch_size 32
```

### Q: 找不到数据文件？
**A**: 确保 `NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv` 存在

---

## 📈 预期结果

### 现实预期
- NBA比赛预测本身就很难
- 专业模型准确率通常在 **55-60%**
- 60%+ 已经是很好的结果
- 不要期望 70%+ (几乎不可能)

### 基准对比
| 方法 | 准确率 |
|------|--------|
| 随机猜测 | 50% |
| 简单规则 (主场优势) | ~52% |
| 基础机器学习 | 53-55% |
| 深度学习 (本项目) | **55-60%** |

---

## 📖 数据说明

### 数据来源
- 2010-2024 NBA常规赛数据
- 包含球队统计、比赛结果等

### 特征工程
- 排除数据泄露特征 (PLUS_MINUS, RANK列)
- 使用滑动窗口避免未来信息
- 标准化处理

详见: `NBA数据字段说明.md`

---

## 🛠️ 进阶优化

### 如果想进一步提升:

1. **特征工程**
   - 添加对手特征
   - 添加主客场信息
   - 添加连胜/连败特征

2. **模型改进**
   - 尝试LSTM/GRU
   - 集成学习
   - 超参数搜索

3. **数据增强**
   - 使用更多历史数据
   - 添加球员伤病信息
   - 添加赛程密度特征

---

## 📝 更新日志

### v2.0 (2024-12-02)
- ✅ 修复训练loss不下降问题
- ✅ 添加类别权重平衡
- ✅ 添加标签平滑和梯度裁剪
- ✅ 改进学习率调度
- ✅ 添加批归一化
- ✅ 添加数据诊断工具
- ✅ 完善训练日志

### v1.0
- 初始Transformer模型实现

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- NBA官方数据
- PyTorch团队
- 开源社区

---

## 📧 联系方式

如有问题或建议，欢迎提Issue或PR。

---

**祝训练顺利! 🏀**
