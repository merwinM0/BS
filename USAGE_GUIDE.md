# NBA Transformer模型使用指南

## 🚀 快速开始

### 1. 安装依赖

项目使用 `uv` 包管理器。如果下载超时，可以增加超时时间：

```bash
# Windows PowerShell
$env:UV_HTTP_TIMEOUT="300"
uv sync

# 或者使用pip直接安装
pip install torch pandas numpy scikit-learn
```

### 2. 训练模型

#### 使用默认参数训练
```bash
uv run python train_transformer.py
```

或者如果已经安装依赖：
```bash
python train_transformer.py
```

#### 使用自定义参数
```bash
python train_transformer.py \
    --window_size 5 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.0005
```

### 3. 在代码中使用

```python
from nba_transformer_predictor import NBATransformerPredictor

# 创建预测器
predictor = NBATransformerPredictor(window_size=3)

# 加载和准备数据
predictor.load_data()
predictor.prepare_features()
predictor.create_sequences(test_year='2023-24')

# 构建模型
predictor.build_model(d_model=128, nhead=8, num_layers=4)

# 训练
predictor.train(epochs=50, batch_size=64, lr=0.001)

# 评估
results = predictor.evaluate()
print(f"测试准确率: {results['accuracy']:.4f}")
```

---

## 📊 模型说明

### 核心概念

1. **滑动窗口**: 使用球队最近N场比赛(默认3场)的数据来预测下一场比赛结果
2. **序列建模**: 每支球队的比赛按时间顺序排列，形成时间序列
3. **Transformer**: 使用多头注意力机制捕捉比赛之间的关系
4. **Softmax输出**: 输出 [P(Loss), P(Win)] 概率分布

### 数据流程

```
原始数据 (CSV)
    ↓
按球队和日期排序
    ↓
提取特征 (18个统计指标)
    ↓
创建滑动窗口序列
    [G1, G2, G3] → 预测 G4
    [G2, G3, G4] → 预测 G5
    ...
    ↓
标准化特征
    ↓
训练Transformer模型
    ↓
输出胜负概率
```

---

## 🎯 参数调优建议

### 提升准确率

1. **增加窗口大小**
   ```bash
   python train_transformer.py --window_size 5
   ```
   使用更多历史比赛信息

2. **增加模型容量**
   ```bash
   python train_transformer.py --d_model 256 --num_layers 6
   ```
   更大的模型可以学习更复杂的模式

3. **调整学习率**
   ```bash
   python train_transformer.py --lr 0.0005 --epochs 100
   ```
   较小的学习率配合更多训练轮数

4. **增加正则化**
   ```bash
   python train_transformer.py --dropout 0.2 --weight_decay 1e-4
   ```
   防止过拟合

### 加快训练速度

1. **减少模型大小**
   ```bash
   python train_transformer.py --d_model 64 --num_layers 2
   ```

2. **增大批次**
   ```bash
   python train_transformer.py --batch_size 128
   ```
   如果内存允许

3. **使用GPU**
   - 模型会自动检测并使用CUDA
   - 确保安装了GPU版本的PyTorch

---

## 📈 预期性能

### 基准性能 (默认参数)
- **训练集准确率**: 65-75%
- **测试集准确率**: 60-70%
- **训练时间**: 
  - CPU: ~5-10分钟
  - GPU: ~1-2分钟

### 性能对比

| 模型 | 窗口大小 | 准确率 | 训练时间 |
|------|----------|--------|----------|
| 基础模型 | 3 | ~62% | 5分钟 |
| 中等模型 | 5 | ~65% | 8分钟 |
| 大型模型 | 7 | ~68% | 12分钟 |

---

## 🔧 故障排除

### 问题1: 依赖安装失败

**症状**: `UV_HTTP_TIMEOUT` 错误

**解决方案**:
```bash
# 增加超时时间
$env:UV_HTTP_TIMEOUT="300"  # Windows
export UV_HTTP_TIMEOUT=300   # Linux/Mac

# 或使用pip
pip install torch pandas numpy scikit-learn
```

### 问题2: CUDA不可用

**症状**: 训练很慢，显示 "使用设备: cpu"

**解决方案**:
- 安装CUDA版本的PyTorch
- 检查NVIDIA驱动是否正确安装
- CPU训练也可以，只是较慢

### 问题3: 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小批次大小
python train_transformer.py --batch_size 32

# 或减小模型大小
python train_transformer.py --d_model 64 --num_layers 2
```

### 问题4: 数据文件未找到

**症状**: `FileNotFoundError: regular_season_totals_2010_2024.csv`

**解决方案**:
- 确保 `NBA-Data-2010-2024-main` 目录存在
- 检查CSV文件是否在正确位置
- 文件路径: `./NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv`

---

## 📝 完整训练示例

```python
from nba_transformer_predictor import NBATransformerPredictor

# 1. 创建预测器
print("创建预测器...")
predictor = NBATransformerPredictor(
    window_size=3,
    model_dir="saved_model_transformer"
)

# 2. 加载数据
print("加载数据...")
predictor.load_data()
# 输出: ✓ 加载数据: 35040 条记录

# 3. 准备特征
print("准备特征...")
predictor.prepare_features()
# 输出: ✓ 选择特征数量: 18

# 4. 创建序列
print("创建序列...")
predictor.create_sequences(test_year='2023-24')
# 输出: ✓ 训练集: 28456 个序列
#       ✓ 测试集: 2134 个序列

# 5. 构建模型
print("构建模型...")
predictor.build_model(
    d_model=128,
    nhead=8,
    num_layers=4,
    dropout=0.1
)
# 输出: ✓ 模型构建完成
#       总参数量: 245,634

# 6. 训练模型
print("训练模型...")
best_acc = predictor.train(
    epochs=50,
    batch_size=64,
    lr=0.001,
    weight_decay=1e-5
)
# 输出: 训练完成! 最佳测试准确率: 0.6823

# 7. 评估模型
print("评估模型...")
results = predictor.evaluate()
print(f"最终准确率: {results['accuracy']:.4f}")

# 8. 保存模型 (训练时自动保存最佳模型)
print("模型已保存到: saved_model_transformer/")
```

---

## 🎮 预测比赛示例

```python
from nba_transformer_predictor import NBATransformerPredictor
import numpy as np

# 加载训练好的模型
predictor = NBATransformerPredictor(window_size=3)
predictor.load_model()

# 准备球队历史数据
# 假设我们有两支球队最近3场比赛的数据
# 每场比赛有18个特征

# 球队1最近3场比赛 (示例数据)
team1_history = np.array([
    [240, 38, 85, 0.447, 12, 35, 0.343, 18, 22, 0.818, 10, 35, 45, 25, 12, 8, 5, 20],  # 第1场
    [240, 42, 88, 0.477, 15, 38, 0.395, 20, 24, 0.833, 12, 33, 45, 28, 10, 9, 6, 18],  # 第2场
    [240, 40, 86, 0.465, 13, 36, 0.361, 19, 23, 0.826, 11, 34, 45, 26, 11, 8, 5, 19],  # 第3场
])

# 球队2最近3场比赛 (示例数据)
team2_history = np.array([
    [240, 35, 82, 0.427, 10, 32, 0.313, 16, 20, 0.800, 9, 32, 41, 22, 14, 7, 4, 15],
    [240, 37, 84, 0.440, 11, 33, 0.333, 17, 21, 0.810, 10, 33, 43, 24, 13, 7, 5, 16],
    [240, 36, 83, 0.434, 10, 31, 0.323, 16, 19, 0.842, 9, 31, 40, 23, 12, 6, 4, 14],
])

# 预测比赛结果
result = predictor.predict_match(team1_history, team2_history)

print(f"球队1胜率: {result['team1_win_prob']:.2%}")
print(f"球队2胜率: {result['team2_win_prob']:.2%}")
print(f"预测获胜: {result['predicted_winner']}")
print(f"置信度: {result['confidence']:.2%}")

# 输出示例:
# 球队1胜率: 58.32%
# 球队2胜率: 41.68%
# 预测获胜: team1
# 置信度: 16.64%
```

---

## 📚 特征说明

模型使用的18个特征:

| 特征 | 说明 | 示例值 |
|------|------|--------|
| MIN | 总分钟数 | 240 |
| FGM | 投篮命中数 | 40 |
| FGA | 投篮出手数 | 85 |
| FG_PCT | 投篮命中率 | 0.471 |
| FG3M | 三分命中数 | 12 |
| FG3A | 三分出手数 | 35 |
| FG3_PCT | 三分命中率 | 0.343 |
| FTM | 罚球命中数 | 18 |
| FTA | 罚球出手数 | 22 |
| FT_PCT | 罚球命中率 | 0.818 |
| OREB | 进攻篮板 | 10 |
| DREB | 防守篮板 | 35 |
| REB | 总篮板 | 45 |
| AST | 助攻 | 25 |
| TOV | 失误 | 12 |
| STL | 抢断 | 8 |
| BLK | 盖帽 | 5 |
| PF | 犯规 | 20 |

---

## 🎓 进阶使用

### 自定义特征

如果想添加更多特征，修改 `prepare_features()` 方法:

```python
# 在 nba_transformer_predictor.py 中
def prepare_features(self):
    # 添加自定义特征
    feature_cols = ['PTS', 'FGM', 'FGA', ...]  # 添加你的特征
    
    # 或者添加计算特征
    self.raw_data['PTS_PER_FGA'] = self.raw_data['PTS'] / self.raw_data['FGA']
    feature_cols.append('PTS_PER_FGA')
```

### 修改测试集

```python
# 使用不同年份作为测试集
predictor.create_sequences(test_year='2022-23')

# 或使用最近N场比赛
predictor.create_sequences(test_year='2023-24')
```

### 集成多个模型

```python
# 训练多个不同配置的模型
models = []
for window_size in [3, 5, 7]:
    predictor = NBATransformerPredictor(window_size=window_size)
    predictor.load_data()
    predictor.prepare_features()
    predictor.create_sequences()
    predictor.build_model()
    predictor.train()
    models.append(predictor)

# 投票预测
def ensemble_predict(models, team1_history, team2_history):
    votes = []
    for model in models:
        result = model.predict_match(team1_history, team2_history)
        votes.append(result['predicted_winner'])
    
    # 多数投票
    from collections import Counter
    winner = Counter(votes).most_common(1)[0][0]
    return winner
```

---

## 📞 联系与反馈

如有问题或建议，请查看:
- 主README: `README.md`
- Transformer说明: `TRANSFORMER_README.md`
- 数据字段说明: `NBA数据字段说明.md`

祝训练顺利！🏀
