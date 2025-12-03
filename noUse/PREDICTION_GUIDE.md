# NBA比赛预测使用指南

## 🚀 快速开始

### 1. 确保模型已训练

如果还没有训练模型，先运行：

```bash
python train_transformer.py
```

训练完成后，模型会保存在 `saved_model_transformer/` 目录。

---

## 🎯 预测比赛

### 方法1: 命令行预测 (推荐)

```bash
# 基本用法
python predict_match.py <球队1> <球队2>

# 示例
python predict_match.py LAL GSW
python predict_match.py Lakers Warriors
python predict_match.py "Los Angeles Lakers" "Golden State Warriors"
```

**输出示例:**
```
======================================================================
🏀 NBA比赛预测系统
======================================================================

📦 加载模型...
✓ 模型加载成功!

📊 加载数据...
✓ 选择特征数量: 45

======================================================================
⚔️  LAL vs GSW
======================================================================

  ✓ LAL 注意力热力图: prediction_output/attention_LAL.png
  ✓ GSW 注意力热力图: prediction_output/attention_GSW.png

✅ 注意力热力图已保存到: prediction_output/

======================================================================
📊 预测结果
======================================================================

⚔️  LAL vs GSW
----------------------------------------------------------------------

   LAL |████████████████████████████████                  | 65.3% 🏆
   GSW |█████████████████                                 | 34.7% 

✨ 预测获胜: LAL
💪 置信度: 30.60%

📈 注意力热力图已保存到: prediction_output/
   - attention_LAL.png
   - attention_GSW.png

======================================================================
✅ 预测完成!
======================================================================
```

### 方法2: 交互式预测

```bash
python predict_interactive.py
```

**交互流程:**
1. 输入球队1名称
2. 输入球队2名称
3. 查看预测结果和注意力热力图
4. 选择继续预测或退出

**特殊命令:**
- 输入 `list` - 查看所有可用球队
- 输入 `quit` - 退出程序

### 方法3: Python代码

```python
from nba_transformer_predictor import NBATransformerPredictor

# 加载模型
predictor = NBATransformerPredictor(window_size=3)
predictor.load_model()
predictor.load_data()
predictor.prepare_features()

# 预测
result = predictor.predict_by_team_names('LAL', 'GSW', save_attention=True)

# 查看结果
print(f"{result['team1_name']} 胜率: {result['team1_win_prob']:.2%}")
print(f"{result['team2_name']} 胜率: {result['team2_win_prob']:.2%}")
print(f"预测获胜: {result['team1_name'] if result['predicted_winner']=='team1' else result['team2_name']}")
```

---

## 📊 注意力热力图说明

### 什么是注意力热力图？

注意力热力图显示了Transformer模型在预测时，对历史比赛的关注程度。

### 如何解读？

```
Layer 1 Attention
        Game 1  Game 2  Game 3
Game 1   0.15    0.20    0.65   ← Game 1 最关注 Game 3
Game 2   0.10    0.30    0.60   ← Game 2 最关注 Game 3
Game 3   0.05    0.15    0.80   ← Game 3 最关注自己
```

**颜色说明:**
- 🟥 **深红色** (高值): 高度关注，该场比赛对预测很重要
- 🟨 **黄色** (中值): 中等关注
- ⬜ **浅色** (低值): 较少关注

**典型模式:**
1. **对角线高亮**: 模型关注当前比赛本身
2. **最后一列高亮**: 模型最关注最近的比赛
3. **分散注意力**: 模型综合考虑多场比赛

### 文件位置

注意力热力图保存在 `prediction_output/` 目录：
```
prediction_output/
├── attention_LAL.png    # 湖人队的注意力图
└── attention_GSW.png    # 勇士队的注意力图
```

---

## 🏀 支持的球队

### 常用球队缩写

| 缩写 | 球队全名 | 示例 |
|------|----------|------|
| LAL | Los Angeles Lakers | `python predict_match.py LAL GSW` |
| GSW | Golden State Warriors | `python predict_match.py GSW LAL` |
| BOS | Boston Celtics | `python predict_match.py BOS MIA` |
| MIA | Miami Heat | `python predict_match.py MIA BOS` |
| CHI | Chicago Bulls | `python predict_match.py CHI DET` |
| NYK | New York Knicks | `python predict_match.py NYK BKN` |
| BKN | Brooklyn Nets | `python predict_match.py BKN NYK` |
| PHI | Philadelphia 76ers | `python predict_match.py PHI BOS` |
| TOR | Toronto Raptors | `python predict_match.py TOR CLE` |
| MIL | Milwaukee Bucks | `python predict_match.py MIL CHI` |

### 查看完整列表

```bash
# 方法1: 使用交互式预测
python predict_interactive.py
# 然后输入 'list'

# 方法2: Python代码
python -c "from nba_transformer_predictor import *; p=NBATransformerPredictor(); p.load_data(); print(sorted(p.raw_data['TEAM_ABBREVIATION'].unique()))"
```

---

## ⚙️ 高级选项

### 自定义输出目录

```bash
python predict_match.py LAL GSW --output-dir my_predictions
```

### 不保存注意力图 (更快)

```bash
python predict_match.py LAL GSW --no-attention
```

### 使用不同的窗口大小

```bash
python predict_match.py LAL GSW --window-size 5
```

**注意**: 窗口大小必须与训练时一致！

---

## 🔍 预测原理

### 模型如何预测？

1. **获取历史数据**: 提取两支球队最近N场比赛的数据
2. **独立预测**: 分别预测每支球队"下一场赢的概率"
3. **比较概率**: 概率高的球队预测获胜
4. **归一化**: 将两个概率归一化，使其和为100%

### 示例

```
湖人队最近3场 → 模型预测 → 下一场赢的概率: 0.68
勇士队最近3场 → 模型预测 → 下一场赢的概率: 0.52

归一化:
湖人: 0.68 / (0.68 + 0.52) = 56.7%
勇士: 0.52 / (0.68 + 0.52) = 43.3%

预测: 湖人队获胜 (置信度: 13.4%)
```

---

## ❓ 常见问题

### Q: 找不到球队？
**A**: 
- 确保拼写正确
- 使用标准缩写 (如 LAL, GSW)
- 输入 `list` 查看所有可用球队

### Q: 预测准确率有多高？
**A**: 
- 模型测试准确率约 55-60%
- 这是NBA预测的专业水平
- 不要期望100%准确

### Q: 为什么有时预测不准？
**A**: 
- NBA比赛受多种因素影响 (伤病、状态、主客场等)
- 模型只基于历史统计数据
- 实际比赛有很多不可预测因素

### Q: 注意力图显示乱码？
**A**: 
- Windows系统可能缺少中文字体
- 图片仍然可以查看，只是标题可能显示不正常
- 主要看热力图的颜色分布

### Q: 预测速度慢？
**A**: 
- 使用 `--no-attention` 跳过热力图生成
- 确保使用GPU (如果可用)
- 第一次预测会慢一些 (加载模型)

---

## 💡 使用技巧

### 1. 批量预测

创建一个脚本预测多场比赛：

```python
from nba_transformer_predictor import NBATransformerPredictor

predictor = NBATransformerPredictor()
predictor.load_model()
predictor.load_data()
predictor.prepare_features()

matches = [
    ('LAL', 'GSW'),
    ('BOS', 'MIA'),
    ('CHI', 'DET'),
]

for team1, team2 in matches:
    result = predictor.predict_by_team_names(team1, team2, save_attention=False)
    winner = result['team1_name'] if result['predicted_winner']=='team1' else result['team2_name']
    print(f"{team1} vs {team2}: {winner} 获胜")
```

### 2. 分析注意力模式

查看哪些历史比赛对预测最重要：

```python
result = predictor.predict_by_team_names('LAL', 'GSW', save_attention=True)

# 获取注意力权重
attn = result['team1_attention']
for i, layer_attn in enumerate(attn):
    print(f"Layer {i+1}:")
    print(layer_attn[0].cpu().numpy())
```

### 3. 保存预测历史

```python
import json
from datetime import datetime

result = predictor.predict_by_team_names('LAL', 'GSW')
result['timestamp'] = datetime.now().isoformat()

# 保存
with open('prediction_history.json', 'a') as f:
    json.dump(result, f)
    f.write('\n')
```

---

## 📈 下一步

- 训练更多轮次提高准确率
- 尝试不同的窗口大小
- 分析注意力模式找出规律
- 结合其他信息 (伤病、主客场) 做综合判断

---

**祝预测顺利! 🏀**
