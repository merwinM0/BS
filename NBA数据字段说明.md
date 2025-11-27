# NBA数据字段说明

## Totals数据文件字段特征

### 基础信息字段
- **SEASON_YEAR**: 赛季年份 (如2022-23)
- **TEAM_ID**: 球队ID
- **TEAM_ABBREVIATION**: 球队缩写 (如GSW, MIL)
- **TEAM_NAME**: 球队全名
- **GAME_ID**: 比赛ID
- **GAME_DATE**: 比赛日期
- **MATCHUP**: 比赛对阵 (如"GSW @ POR")
- **WL**: 比赛结果 (W/L)

### 比赛统计字段
- **MIN**: 总分钟数
- **FGM**: 投篮命中数
- **FGA**: 投篮出手数
- **FG_PCT**: 投篮命中率
- **FG3M**: 三分命中数
- **FG3A**: 三分出手数
- **FG3_PCT**: 三分命中率
- **FTM**: 罚球命中数
- **FTA**: 罚球出手数
- **FT_PCT**: 罚球命中率
- **OREB**: 进攻篮板
- **DREB**: 防守篮板
- **REB**: 总篮板
- **AST**: 助攻
- **TOV**: 失误
- **STL**: 抢断
- **BLK**: 盖帽
- **BLKA**: 被盖帽
- **PF**: 个人犯规
- **PFD**: 造成犯规
- **PTS**: 得分
- **PLUS_MINUS**: 正负值

### 排名字段
- **GP_RANK**: 场次排名
- **W_RANK**: 胜场排名
- **L_RANK**: 负场排名
- **W_PCT_RANK**: 胜率排名
- **MIN_RANK**: 分钟排名
- **FGM_RANK, FGA_RANK, FG_PCT_RANK**: 投篮相关排名
- **FG3M_RANK, FG3A_RANK, FG3_PCT_RANK**: 三分相关排名
- **FTM_RANK, FTA_RANK, FT_PCT_RANK**: 罚球相关排名
- **OREB_RANK, DREB_RANK, REB_RANK**: 篮板相关排名
- **AST_RANK, TOV_RANK, STL_RANK, BLK_RANK, BLKA_RANK**: 其他统计排名
- **PF_RANK, PFD_RANK, PTS_RANK, PLUS_MINUS_RANK**: 犯规、得分、正负值排名

### 状态字段
- **AVAILABLE_FLAG**: 数据可用性标志

## 数据文件说明

### 两个Totals文件的区别
- **regular_season_totals_2010_2024.csv**: 常规赛数据
- **play_off_totals_2010_2024.csv**: 季后赛数据

字段结构完全一致，只是数据来源不同（常规赛vs季后赛）。

## Box Score与Totals数据的主要区别

### Box Score CSV
- 记录每场比赛中每个球员的详细表现
- 包含球员信息：`personId`, `personName`, `position`, `jerseyNum`
- 包含单场比赛数据：`minutes`, `fieldGoalsMade`, `points`, `assists`等
- 每行代表一个球员在一场比赛中的表现

### Totals CSV
- 记录每支球队在每场比赛中的总计数据
- 包含球队信息：`TEAM_ID`, `TEAM_ABBREVIATION`, `TEAM_NAME`
- 包含比赛结果：`WL` (胜负), `MIN` (总分钟数)
- 包含球队统计：`FGM`, `FGA`, `REB`, `AST`等总计数据
- 每行代表一支球队在一场比赛中的整体表现
- 额外包含排名信息：各种`_RANK`字段

简单说，Box Score是球员级别的详细数据，Totals是球队级别的汇总数据。