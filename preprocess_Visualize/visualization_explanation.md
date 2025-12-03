# 数据可视化说明

## 操作内容

1. **数据初步分析区分分类变量与连续型变量（并绘图进行可视化分析）**：
   - 分类变量：SEASON_YEAR, TEAM_ABBREVIATION, TEAM_NAME, GAME_DATE, MATCHUP, WL
   - 连续变量：TEAM_ID, GAME_ID, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, TOV, STL, BLK, BLKA, PF, PFD, PTS, PLUS_MINUS, GP_RANK, W_RANK, L_RANK, W_PCT_RANK, MIN_RANK, FGM_RANK, FGA_RANK, FG_PCT_RANK, FG3M_RANK, FG3A_RANK, FG3_PCT_RANK, FTM_RANK, FTA_RANK, FT_PCT_RANK, OREB_RANK, DREB_RANK, REB_RANK, AST_RANK, TOV_RANK, STL_RANK, BLK_RANK, BLKA_RANK, PF_RANK, PFD_RANK, PTS_RANK, PLUS_MINUS_RANK, AVAILABLE_FLAG, PTS_discretized
   - 可视化：对关键分类变量WL绘制条形图，对关键连续变量PTS和AST绘制直方图和箱线图。

2. **变量相关性分析（可视化）**：
   - 计算连续变量的相关矩阵，绘制热力图。

3. **特征选择**：
   - 使用SelectKBest（f_regression）选择与PTS最相关的10个特征：FGM, FG_PCT, FG3M, AST, PLUS_MINUS, FGM_RANK, FG_PCT_RANK, PTS_RANK, PLUS_MINUS_RANK, PTS_discretized
   - 绘制特征选择得分条形图。

4. **特征提取**：
   - 使用PCA将连续变量降维到2个主成分，绘制散点图。

5. **特征编码**：
   - 对分类变量使用LabelEncoder进行编码，新增编码列。

图表保存到imgoutput/文件夹，标签使用中文，字体为Noto Sans CJK JP。

编码后的数据保存到output/encoded_data.csv。