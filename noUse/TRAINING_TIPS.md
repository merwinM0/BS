# NBA模型训练优化指南

## 已实施的改进

### 1. **降低学习率** (0.001 → 0.0001)
- 原因: 学习率过高导致loss震荡,无法收敛
- 效果: 更稳定的训练过程

### 2. **添加梯度裁剪** (max_norm=1.0)
- 原因: 防止梯度爆炸
- 效果: 训练更稳定

### 3. **类别权重平衡**
- 原因: 如果胜负比例不均衡,模型会偏向多数类
- 效果: 自动计算并应用类别权重

### 4. **标签平滑** (label_smoothing=0.1)
- 原因: 防止过拟合,提高泛化能力
- 效果: 模型不会过度自信

### 5. **改进学习率调度**
- 从 ReduceLROnPlateau → CosineAnnealingWarmRestarts
- 效果: 更好的学习率衰减策略

### 6. **批归一化 (Batch Normalization)**
- 在全连接层添加BN
- 效果: 加速训练,提高稳定性

### 7. **Xavier权重初始化**
- 使用xavier_uniform初始化
- 效果: 更好的初始权重分布

### 8. **早停机制** (patience=15)
- 15轮无改善自动停止
- 效果: 防止过拟合,节省时间

### 9. **详细训练日志**
- 每轮显示: TrainLoss, TestLoss, TrainAcc, TestAcc, LR
- 效果: 更好地监控训练过程

## 训练命令

### 基础训练 (使用默认优化参数)
```bash
python train_transformer.py
```

### 自定义参数训练
```bash
python train_transformer.py --epochs 100 --batch_size 32 --lr 0.0001 --window_size 5
```

### 完整参数示例
```bash
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

## 参数调优建议

### 如果Loss不下降:
1. **降低学习率**: `--lr 0.00005` 或 `--lr 0.00001`
2. **增加批次大小**: `--batch_size 128`
3. **减少模型复杂度**: `--num_layers 2 --d_model 64`
4. **检查数据**: 确保特征没有泄露信息

### 如果准确率只有50%左右:
1. **增加窗口大小**: `--window_size 5` 或 `--window_size 7`
2. **增加模型容量**: `--d_model 256 --num_layers 6`
3. **降低正则化**: `--dropout 0.05 --weight_decay 1e-5`
4. **增加训练轮数**: `--epochs 200`

### 如果过拟合 (训练准确率高,测试准确率低):
1. **增加正则化**: `--dropout 0.3 --weight_decay 1e-3`
2. **增加标签平滑**: `--label_smoothing 0.2`
3. **减少模型容量**: `--d_model 64 --num_layers 2`
4. **使用更多数据**: 减少test_year范围

### 如果训练太慢:
1. **减少模型大小**: `--d_model 64 --num_layers 2`
2. **增加批次大小**: `--batch_size 128`
3. **减少窗口大小**: `--window_size 3`

## 监控指标

### 健康的训练应该显示:
- ✅ TrainLoss 持续下降
- ✅ TestLoss 下降或稳定
- ✅ TrainAcc 和 TestAcc 都在提升
- ✅ TestAcc 最终 > 55%

### 问题信号:
- ❌ Loss 不变或上升 → 学习率太高或数据有问题
- ❌ TrainAcc 高但 TestAcc 低 → 过拟合
- ❌ 两者都很低 (~50%) → 模型容量不足或特征不够好
- ❌ Loss 变成 NaN → 梯度爆炸,降低学习率

## 数据质量检查

### 确保没有数据泄露:
- ✅ 已排除 `PLUS_MINUS` (这是比赛结果的直接指标)
- ✅ 已排除所有 `_RANK` 列 (赛季累计排名)
- ✅ 使用滑动窗口,只看历史数据

### 特征工程建议:
1. 使用滚动平均而非累计统计
2. 添加对手特征
3. 添加主客场信息
4. 添加连胜/连败特征
5. 添加球员伤病信息

## 预期结果

### 现实预期:
- NBA比赛预测本身就很难
- 专业模型准确率通常在 55-60%
- 60%+ 已经是很好的结果
- 不要期望 70%+ (几乎不可能)

### 基准对比:
- 随机猜测: 50%
- 简单规则 (主场优势): ~52%
- 基础机器学习: 53-55%
- 深度学习: 55-60%
- **目标: 56-58%**

## 故障排除

### 如果仍然无法训练:
1. 检查CUDA是否可用: 模型会自动使用GPU
2. 检查数据文件路径: `./NBA-Data-2010-2024-main/`
3. 检查Python版本: 需要 Python 3.8+
4. 检查PyTorch版本: `pip install torch --upgrade`

### 获取更多信息:
训练时会自动显示:
- 类别分布 (胜/负比例)
- 类别权重
- 模型参数量
- 每轮详细指标

## 下一步优化方向

1. **特征工程**: 添加更多有意义的特征
2. **对手建模**: 同时考虑对手的历史表现
3. **时序建模**: 使用LSTM或GRU替代Transformer
4. **集成学习**: 训练多个模型并投票
5. **超参数搜索**: 使用Optuna自动调参
