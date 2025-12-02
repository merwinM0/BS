# 项目结构说明

## 📁 当前项目结构 (已整理)

```
BS/
├── 📄 核心文档 (3个)
│   ├── README.md                      # 主文档 (整合版)
│   ├── TRAINING_TIPS.md               # 训练优化指南
│   └── NBA数据字段说明.md              # 数据字段文档
│
├── 🐍 核心Python文件 (7个)
│   ├── nba_transformer_predictor.py   # ⭐ 主模型 (Transformer)
│   ├── train_transformer.py           # ⭐ 训练脚本
│   ├── diagnose_data.py               # ⭐ 数据诊断工具
│   ├── test_transformer_quick.py      # 测试脚本
│   ├── data_preprocessor.py           # 数据预处理
│   ├── nba_data_analyzer.py           # 数据分析
│   └── main.py                        # 主入口
│
├── 📊 数据目录
│   ├── NBA-Data-2010-2024-main/       # 原始数据
│   ├── processed_data/                # 处理后数据
│   ├── saved_model_transformer/       # 模型保存
│   ├── prediction_output/             # 预测输出
│   └── analysis_output/               # 分析输出
│
├── 📦 配置文件
│   ├── pyproject.toml                 # 依赖配置
│   └── uv.lock                        # 锁定文件
│
└── 🗂️ 旧版本归档
    └── old_predictor/                 # 旧代码和文档
        ├── nba_predictor.py           # 旧版预测器
        ├── nba_predictor_mini.py
        ├── nba_model_predictor.py
        ├── nba_full_model_predictor.py
        ├── README.md                  # 旧版文档
        ├── QUICK_START.md
        ├── USAGE_GUIDE.md
        ├── TRANSFORMER_README.md
        ├── TRANSFORMER_SUMMARY.md
        └── README_OLD_FILES.md        # 归档说明
```

## ✅ 整理完成的工作

### 1. 文档整合
- ✅ 将5个分散的MD文件整合为1个主README
- ✅ 保留必要的专项文档 (TRAINING_TIPS, 数据字段说明)
- ✅ 移除重复和过时的文档

### 2. 代码整理
- ✅ 将4个旧版预测器移至 `old_predictor/`
- ✅ 保留当前使用的核心文件
- ✅ 清晰的文件命名和组织

### 3. 目录结构
- ✅ 根目录只保留当前使用的文件
- ✅ 旧版本统一归档到 `old_predictor/`
- ✅ 添加归档说明文档

## 🎯 当前使用的核心文件

### 必读文档
1. **README.md** - 完整的项目文档
   - 快速开始
   - 模型架构
   - 参数说明
   - 使用示例
   - 常见问题

2. **TRAINING_TIPS.md** - 训练优化指南
   - 9项关键改进
   - 参数调优建议
   - 故障排除
   - 预期结果

3. **NBA数据字段说明.md** - 数据字段文档

### 核心代码
1. **nba_transformer_predictor.py** - 主模型实现
2. **train_transformer.py** - 训练脚本
3. **diagnose_data.py** - 数据诊断工具

## 🚀 快速开始

```bash
# 1. 诊断数据
python diagnose_data.py

# 2. 训练模型
python train_transformer.py

# 3. 查看文档
# 阅读 README.md 和 TRAINING_TIPS.md
```

## 📝 文件数量对比

### 整理前
- MD文档: 7个 (分散、重复)
- Python文件: 11个 (新旧混杂)

### 整理后
- MD文档: 3个 (精简、清晰)
- Python文件: 7个 (当前使用)
- 归档文件: 10个 (old_predictor/)

## 🎉 整理效果

- ✅ 根目录清爽，只保留当前使用的文件
- ✅ 文档整合，避免信息分散
- ✅ 旧版本归档，便于查找历史
- ✅ 结构清晰，易于维护

---

**整理完成时间**: 2024-12-02
