# NBA Transformer预测模型 - 快速开始

## 📦 文件清单

确保你有以下文件：
- ✅ `nba_transformer_predictor.py` - 主模型
- ✅ `train_transformer.py` - 训练脚本
- ✅ `pyproject.toml` - 依赖配置
- ✅ `NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv` - 数据文件

## 🚀 三步开始

### 1. 安装依赖
```bash
pip install torch pandas numpy scikit-learn
```

### 2. 训练模型
```bash
python train_transformer.py
```

### 3. 查看结果
训练完成后，模型会保存在 `saved_model_transformer/` 目录。

## 📊 预期结果
- 训练时间: 5-10分钟 (CPU)
- 测试准确率: 60-70%
- 模型参数: ~245,000

## 📚 详细文档
- `USAGE_GUIDE.md` - 完整使用指南
- `TRANSFORMER_README.md` - 技术文档
- `TRANSFORMER_SUMMARY.md` - 项目总结

## ⚙️ 自定义参数
```bash
# 使用5场比赛预测
python train_transformer.py --window_size 5

# 训练更多轮次
python train_transformer.py --epochs 100

# 使用更大的批次
python train_transformer.py --batch_size 128
```

## 🎯 模型特点
- ✅ Transformer架构 (4层，8个注意力头)
- ✅ 滑动窗口 (默认3场比赛预测下一场)
- ✅ Softmax概率输出
- ✅ 2023-24赛季作为测试集

## 💡 在代码中使用
```python
from nba_transformer_predictor import train_and_save

# 训练模型
predictor = train_and_save()

# 评估
results = predictor.evaluate()
print(f"准确率: {results['accuracy']:.2%}")
```

## ❓ 常见问题

**Q: 依赖安装失败？**  
A: 尝试 `pip install torch pandas numpy scikit-learn` 单独安装

**Q: 找不到数据文件？**  
A: 确保 `NBA-Data-2010-2024-main/regular_season_totals_2010_2024.csv` 存在

**Q: 训练太慢？**  
A: 使用 `--d_model 64 --num_layers 2` 减小模型

**Q: 内存不足？**  
A: 使用 `--batch_size 32` 减小批次大小

---

更多信息请查看 `USAGE_GUIDE.md`
