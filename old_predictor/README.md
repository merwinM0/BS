# NBA 比赛结果预测系统

> 基于机器学习的 NBA 比赛胜负预测 —— 数据挖掘与分析实践项目

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目概述

本项目是一个完整的**数据挖掘与分析实践项目**，通过对 2010-2024 年 NBA 比赛数据的深度分析和机器学习建模，实现对 NBA 比赛结果的智能预测。项目涵盖了数据采集、数据清洗、特征工程、模型训练、模型评估等数据挖掘全流程。

### 🎯 项目目标

- 构建高准确率的 NBA 比赛胜负预测模型
- 探索影响比赛结果的关键因素
- 实践数据挖掘与机器学习的完整工作流程
- 提供可交互的预测工具

---

## 🏗️ 项目架构

```
BS-master/
├── NBA-Data-2010-2024-main/          # 原始数据目录
│   ├── regular_season_totals_2010_2024.csv      # 常规赛球队统计 (~9MB)
│   ├── play_off_totals_2010_2024.csv            # 季后赛球队统计 (~577KB)
│   ├── regular_season_box_scores_*.csv          # 常规赛球员详细数据 (~67MB)
│   └── play_off_box_scores_2010_2024.csv        # 季后赛球员详细数据 (~4.5MB)
├── processed_data/                   # 预处理后的数据
├── saved_model/                      # 基础模型存储
├── saved_model_full/                 # 完整模型存储
├── data_preprocessor.py              # 数据预处理模块
├── nba_data_analyzer.py              # 数据分析模块
├── nba_model_predictor.py            # 基础预测模型
├── nba_full_model_predictor.py       # 完整预测模型（使用全部数据）
├── NBA数据字段说明.md                 # 数据字段文档
└── README.md                         # 项目说明文档
```

---

## 🔬 技术原理

### 1. 数据预处理 (Data Preprocessing)

数据预处理是数据挖掘的基础环节，本项目采用以下技术：

#### 1.1 数据清洗
- **缺失值处理**：采用均值填充策略处理数值型缺失值
- **异常值检测**：基于 IQR (四分位距) 方法识别和处理异常值
- **数据类型转换**：统一日期格式、数值类型

#### 1.2 特征工程
- **特征选择**：从 50+ 原始特征中筛选出 17 个核心特征
- **特征标准化**：使用 `StandardScaler` 进行 Z-score 标准化
  ```
  z = (x - μ) / σ
  ```
- **标签编码**：将胜负结果 (W/L) 转换为二进制标签 (1/0)

### 2. 机器学习模型

#### 2.1 逻辑回归 (Logistic Regression)

逻辑回归是二分类问题的经典算法，通过 Sigmoid 函数将线性组合映射到概率空间：

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

**优点**：
- 可解释性强，能够分析各特征对结果的影响
- 训练速度快，适合大规模数据
- 输出概率值，便于理解预测置信度

#### 2.2 随机森林 (Random Forest)

随机森林是集成学习方法，通过构建多棵决策树并投票得出最终结果：

```
最终预测 = Mode(Tree₁, Tree₂, ..., Treeₙ)
```

**核心思想**：
- **Bagging**：有放回抽样构建多个训练子集
- **特征随机性**：每棵树只使用部分特征
- **投票机制**：综合多棵树的预测结果

#### 2.3 梯度提升 (Gradient Boosting)

梯度提升通过迭代地训练弱学习器，每次拟合前一轮的残差：

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

其中 $h_m(x)$ 是第 m 轮训练的弱学习器，$\gamma_m$ 是学习率。

#### 2.4 集成模型 (Ensemble)

本项目的完整版模型采用集成策略，综合三种模型的预测结果：

```python
最终概率 = (P_logistic + P_random_forest + P_gradient_boosting) / 3
```

### 3. 预测原理

预测两支球队比赛结果的流程：

```
输入: 球队A, 球队B
    ↓
获取历史统计数据 (球队A平均数据, 球队B平均数据)
    ↓
特征标准化
    ↓
模型预测 → 获得各自胜率概率 P(A), P(B)
    ↓
相对胜率计算: Win_A = P(A) / (P(A) + P(B))
    ↓
输出: 预测获胜方及置信度
```

---

## 📊 数据说明

### 数据来源
- **时间跨度**：2010-2024 赛季
- **数据规模**：33,000+ 场比赛记录，400,000+ 球员表现记录
- **覆盖范围**：NBA 全部 30 支球队

### 核心特征

| 特征 | 说明 | 类型 |
|------|------|------|
| PTS | 得分 | 数值型 |
| FGM / FGA / FG_PCT | 投篮命中数/出手数/命中率 | 数值型 |
| FG3M / FG3A / FG3_PCT | 三分命中数/出手数/命中率 | 数值型 |
| FTM / FTA / FT_PCT | 罚球命中数/出手数/命中率 | 数值型 |
| OREB / DREB / REB | 进攻篮板/防守篮板/总篮板 | 数值型 |
| AST | 助攻 | 数值型 |
| TOV | 失误 | 数值型 |
| STL | 抢断 | 数值型 |
| BLK | 盖帽 | 数值型 |

### 可预测球队

```
ATL, BKN, BOS, CHA, CHI, CLE, DAL, DEN, DET, GSW, 
HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, 
OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS
```

---

## 💻 硬件需求

### 最低配置

| 组件 | 要求 |
|------|------|
| **CPU** | 双核处理器 (Intel i3 / AMD Ryzen 3 或同等) |
| **内存** | 4 GB RAM |
| **存储** | 500 MB 可用空间 |
| **操作系统** | Windows 10 / macOS 10.14 / Ubuntu 18.04 或更高 |

### 推荐配置

| 组件 | 要求 |
|------|------|
| **CPU** | 四核处理器 (Intel i5 / AMD Ryzen 5 或更高) |
| **内存** | 8 GB RAM 或更高 |
| **存储** | 1 GB SSD 可用空间 |
| **操作系统** | Windows 11 / macOS 12 / Ubuntu 22.04 |

### 性能参考

| 操作 | 最低配置耗时 | 推荐配置耗时 |
|------|-------------|-------------|
| 数据加载 | ~30 秒 | ~10 秒 |
| 模型训练 (基础版) | ~2 分钟 | ~30 秒 |
| 模型训练 (完整版) | ~5 分钟 | ~1 分钟 |
| 单次预测 | <1 秒 | <0.5 秒 |

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd BS-master

# 使用 uv 安装依赖 (推荐)
uv sync

# 或使用 pip
pip install pandas numpy scikit-learn
```

### 依赖列表

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### 使用方式

#### 方式一：命令行预测

```bash
# 基础模型
python nba_model_predictor.py LAL GSW

# 完整模型 (使用全部数据)
python nba_full_model_predictor.py LAL BOS
```

#### 方式二：交互式预测

```bash
python nba_model_predictor.py
# 或
python nba_full_model_predictor.py
```

#### 方式三：代码调用

```python
from nba_full_model_predictor import load_and_predict

# 加载模型
predictor = load_and_predict()

# 预测比赛
result = predictor.predict("LAL", "GSW")
print(f"预测获胜: {result['predicted_winner']}")
print(f"湖人胜率: {result['team1_win_prob']:.1%}")
print(f"勇士胜率: {result['team2_win_prob']:.1%}")
```

### 输出示例

```
🏀 LAL vs GSW
   LAL 胜率: 45.3%
   GSW 胜率: 54.7%
   预测获胜: GSW 🏆
   置信度: 9.4%
```

---

## 📈 模型性能

| 模型 | 准确率 | 训练时间 |
|------|--------|----------|
| 逻辑回归 | ~65% | ~5s |
| 随机森林 | ~67% | ~15s |
| 梯度提升 | ~68% | ~20s |
| 集成模型 | ~69% | ~40s |

> 注：实际准确率受数据分布和随机种子影响，上述数值为参考值。

---

## 🎓 项目意义

### 1. 学术价值

- **数据挖掘实践**：完整展示了从原始数据到预测模型的全流程
- **机器学习应用**：对比了多种分类算法在体育预测领域的表现
- **特征工程探索**：分析了篮球比赛中影响胜负的关键因素

### 2. 技术价值

- **模块化设计**：数据处理、模型训练、预测服务解耦
- **可扩展架构**：易于添加新的数据源和模型
- **工程实践**：包含完整的数据验证、模型持久化、错误处理

### 3. 应用价值

- **体育分析**：为球迷和分析师提供数据驱动的比赛预测
- **决策支持**：辅助理解球队实力对比
- **教育示范**：作为数据科学课程的实践案例

### 4. 技术栈总结

| 领域 | 技术 |
|------|------|
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn |
| 数据标准化 | StandardScaler, MinMaxScaler |
| 模型算法 | Logistic Regression, Random Forest, Gradient Boosting |
| 模型持久化 | Pickle, JSON |

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `data_preprocessor.py` | 数据预处理模块，包含数据清洗、特征工程 |
| `nba_data_analyzer.py` | 数据分析模块，用于探索性数据分析 |
| `nba_model_predictor.py` | 基础预测模型，使用常规赛数据 |
| `nba_full_model_predictor.py` | 完整预测模型，使用全部数据和集成学习 |
| `NBA数据字段说明.md` | 数据字段详细说明文档 |

---

## 🔮 未来改进方向

1. **引入更多特征**
   - 球员伤病信息
   - 主客场因素
   - 背靠背比赛疲劳度
   - 赛季阶段 (常规赛初期/中期/末期)

2. **模型优化**
   - 深度学习模型 (LSTM 处理时序数据)
   - 更精细的超参数调优
   - 交叉验证策略优化

3. **功能扩展**
   - Web API 服务
   - 实时数据更新
   - 可视化分析面板

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

---

<p align="center">
  <i>数据驱动，智能预测 —— NBA 比赛结果预测系统</i>
</p>
