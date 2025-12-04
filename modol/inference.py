import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=8,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 2)

    def forward(self, x, return_attention=False):
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        if return_attention:
            attention_weights = []
            for layer in self.transformer_encoder.layers:
                x_input = x
                attn_output, attn_w = layer.self_attn(x, x, x, need_weights=True,
                                                       average_attn_weights=True)
                x = x_input + layer.dropout1(attn_output)
                x = layer.norm1(x)
                # 前馈部分
                x = layer.norm2(x + layer.dropout2(
                    layer.activation(layer.linear2(layer.dropout(layer.linear1(x))))))
                attention_weights.append(attn_w)
        else:
            x = self.transformer_encoder(x)
            attention_weights = None

        x = x[:, -1, :]  # 取最后一个时间步
        x = self.fc2(self.dropout(self.relu(self.bn1(self.fc1(x)))))
        if return_attention:
            return x, attention_weights
        return x


# ------------------ 推理器 ------------------
class NBAInference:
    def __init__(self, model_dir: str = "./Models",
                 data_path: str = "./Data/play_off_totals_2010_2024.csv"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        # 加载 csv 并保存为实例属性
        self.df = pd.read_csv(self.data_path)
        if "GAME_DATE" in self.df.columns:
            self.df["GAME_DATE"] = pd.to_datetime(self.df["GAME_DATE"])
        print("✅ 模型加载完成")

        with open(self.model_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.feature_names = cfg["feature_names"]
        self.window_size = cfg["window_size"]
        self.d_model = cfg.get("d_model", 128)
        self.nhead = cfg.get("nhead", 8)
        self.num_layers = cfg.get("num_layers", 4)
        self.dropout = cfg.get("dropout", 0.1)

        with open(self.model_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.model = TransformerPredictor(
            feature_dim=len(self.feature_names),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_dir / "transformer_model.pth",
                       map_location=self.device))
        self.model.eval()
        print("✅ 模型加载完成")

    # ---------- 工具 ----------
    def _find_team(self, name: str):
        name = name.upper()
        if name in self.df["TEAM_ABBREVIATION"].values:
            return name
        for abbr in self.df["TEAM_ABBREVIATION"].unique():
            full = self.df[self.df["TEAM_ABBREVIATION"] == abbr]["TEAM_NAME"].iloc[0]
            if name in full.upper():
                return abbr
        return None

    def _get_last_n(self, abbr: str, n: int, before_date=None):
        sub = self.df[self.df["TEAM_ABBREVIATION"] == abbr].copy()
        if before_date is not None:
            sub = sub[sub["GAME_DATE"] <= before_date]
        sub = sub.sort_values("GAME_DATE", ascending=False)
        if len(sub) < n:
            return None
        feats = (sub.head(n)
                 .sort_values("GAME_DATE", ascending=True)[self.feature_names]
                 .fillna(0).values)
        return feats

    def list_real_matchups(self):
        tmp = (self.df.assign(
            away_abbr=self.df["MATCHUP"].str.split().str[-1],
            home_abbr=self.df["TEAM_ABBREVIATION"]
        )[["home_abbr", "away_abbr", "GAME_DATE"]].drop_duplicates()
               .sort_values("GAME_DATE", ascending=False)
               .reset_index(drop=False))
        return tmp

    # ---------- 预测 + 注意力图 ----------
    def predict_with_attn(self, home_abbr: str, away_abbr: str, game_date, orig_idx):
        try:
            row = self.df.loc[orig_idx]
        except KeyError:
            print("❌ 原 csv 索引失效")
            return
        true_label = 1 if row["WL"] == "W" else 0

        h_seq = self._get_last_n(home_abbr, self.window_size, before_date=game_date)
        a_seq = self._get_last_n(away_abbr, self.window_size, before_date=game_date)
        if h_seq is None or a_seq is None:
            print("❌ 某队历史不足，无法预测")
            return

        combo = np.concatenate([h_seq, a_seq], axis=0)
        combo_scaled = self.scaler.transform(combo).reshape(1, 2 * self.window_size, -1)
        combo_tensor = torch.tensor(combo_scaled, dtype=torch.float32).to(self.device)

        # 打开注意力开关
        with torch.no_grad():
            logits, attn_weights = self.model(combo_tensor, return_attention=True)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred_label = 1 if prob > 0.5 else 0

        print("\n***** 预测 + 核对 *****")
        print(f"比赛: {home_abbr} vs {away_abbr}  ({game_date.strftime('%Y-%m-%d')})")
        print(f"真实结果: {'主队胜' if true_label else '主队负'}")
        print(f"模型预测: {'主队胜' if pred_label else '主队负'}  (胜率{prob:.1%})")
        print("预测是否正确: ✅" if pred_label == true_label else "预测是否正确: ❌")

        # 保存注意力图
        self._save_attention_heatmap(
            attn_weights, home_abbr, away_abbr, game_date)

    def _save_attention_heatmap(self, attn_weights, home, away, date):
        save_dir = Path("attention_maps")
        save_dir.mkdir(exist_ok=True)
        file_name = f"{home}_vs_{away}_{date.strftime('%Y%m%d')}.png"
        save_path = save_dir / file_name

        num_layers = len(attn_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6), dpi=150)
        if num_layers == 1:
            axes = [axes]

        axis_labels = [f"{home}(-{i})" for i in range(self.window_size, 0, -1)] + \
                      [f"{away}(-{i})" for i in range(self.window_size, 0, -1)]

        for i, attn in enumerate(attn_weights):
            attn_matrix = attn[0].cpu().numpy()
            ax = axes[i]
            sns.heatmap(attn_matrix,
                        annot=False, fmt=".2f", cmap="YlOrRd",
                        square=True, xticklabels=axis_labels, yticklabels=axis_labels, ax=ax)
            ax.set_title(f"Layer {i+1} Attention")
            ax.set_xlabel("Key (Past Games)")
            ax.set_ylabel("Query (Past Games)")

        plt.suptitle(f"{home} vs {away}  {date.strftime('%Y-%m-%d')}  Attention Heatmap",
                     fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 注意力热点图已保存：{save_path}")



# ------------------ CLI ------------------
def main():
    infer = NBAInference()
    match_df = infer.list_real_matchups()
    if match_df.empty:
        print("❌ csv 里没有任何可识别的对战记录")
        return

    total = len(match_df)
    print(f"季后赛真实对战列表（最新在前，共 {total} 场）：")
    for idx, row in match_df.iterrows():
        print(f"{idx:3d}  {row['home_abbr']} vs {row['away_abbr']}  {row['GAME_DATE'].strftime('%Y-%m-%d')}")

    while True:
        try:
            choice = int(input(f"\n请输入 0-{total-1} 之间的序号（或 Ctrl+C 退出）："))
            if 0 <= choice < total:
                break
            print("序号超出范围，请重新输入！")
        except ValueError:
            print("请输入有效数字！")

    selected = match_df.iloc[choice]
    infer.predict_with_attn(selected["home_abbr"],
                            selected["away_abbr"],
                            selected["GAME_DATE"],
                            selected["index"])


if __name__ == "__main__":
    main()
