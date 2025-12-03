# 旧版本文件说明

本目录包含项目的旧版本代码和文档，已被新版本替代。

## 📁 目录内容

### 旧版Python代码
- `nba_predictor.py` - 旧版基础预测器
- `nba_predictor_mini.py` - 精简版预测器
- `nba_model_predictor.py` - 旧版模型预测器
- `nba_full_model_predictor.py` - 完整版模型预测器

### 旧版文档
- `README.md` - 旧版主文档
- `QUICK_START.md` - 旧版快速开始指南
- `USAGE_GUIDE.md` - 旧版使用指南
- `TRANSFORMER_README.md` - 旧版Transformer文档
- `TRANSFORMER_SUMMARY.md` - 旧版项目总结

## 🔄 替代方案

这些文件已被以下新版本替代:

### 当前使用的文件
- **主代码**: `../nba_transformer_predictor.py` (优化后的Transformer模型)
- **训练脚本**: `../train_transformer.py`
- **主文档**: `../README.md` (整合后的完整文档)
- **训练指南**: `../TRAINING_TIPS.md` (训练优化指南)
- **数据诊断**: `../diagnose_data.py` (新增工具)

## ⚠️ 注意

这些旧文件保留仅供参考，**不建议使用**。

新版本包含以下改进:
- ✅ 修复了训练loss不下降的问题
- ✅ 添加了类别权重平衡
- ✅ 添加了梯度裁剪和标签平滑
- ✅ 改进了学习率调度
- ✅ 添加了批归一化
- ✅ 完善了训练监控

## 📅 归档日期

2024-12-02

---

**请使用项目根目录的最新版本文件！**
