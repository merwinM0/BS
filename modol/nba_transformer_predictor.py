import pandas as pd
import numpy as np
import os
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')



class NBAGameDataset(Dataset):
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TransformerPredictor(nn.Module):
 
    def __init__(self, feature_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层 - 添加批归一化
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 2)  # 二分类: 胜/负
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        if return_attention:
            # 手动遍历每一层以获取注意力权重
            attention_weights = []
            for layer in self.transformer_encoder.layers:
                # 保存输入用于残差连接
                x_input = x
                # 多头注意力
                attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
                x = x_input + layer.dropout1(attn_output)
                x = layer.norm1(x)
                # 前馈网络
                x = layer.norm2(x + layer.dropout2(layer.activation(layer.linear2(layer.dropout(layer.linear1(x))))))
                attention_weights.append(attn_weights)
        else:
            x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
            attention_weights = None
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # (batch, d_model)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 2)
        
        if return_attention:
            return x, attention_weights
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NBATransformerPredictor:
    def __init__(self, model_dir: str = "Models", window_size: int = 3):
        self.model_dir = model_dir
        self.window_size = window_size
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据文件路径
        self.data_dir = './Data'
        
        print(f"使用设备: {self.device}")
        print(f"滑动窗口大小: {window_size}")
    
    def load_data(self):
        print("="*60)
        print("加载NBA数据...")
        print("="*60)
        
        # 加载常规赛球队统计
        regular_file = os.path.join(self.data_dir, 'regular_season_totals_2010_2024.csv')
        df = pd.read_csv(regular_file)
        
        print(f"✓ 加载数据: {len(df)} 条记录")
        print(f"  赛季范围: {df['SEASON_YEAR'].min()} - {df['SEASON_YEAR'].max()}")
        print(f"  球队数量: {df['TEAM_ABBREVIATION'].nunique()}")
        
        # 确保有日期列
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
        else:
            df = df.sort_values(['TEAM_ABBREVIATION', 'SEASON_YEAR'])
        
        self.raw_data = df
        return self
    
    def prepare_features(self):
        print("\n准备特征...")
        
        # 选择特征列(排除泄露特征和非数值特征)
        exclude_cols = [
            'SEASON_YEAR', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL',
            'PLUS_MINUS', 'PLUS_MINUS_RANK',  # 泄露特征
            'AVAILABLE_FLAG'
        ]
        
        # 排除所有RANK列(这些是赛季累计排名,会泄露信息)
        rank_cols = [col for col in self.raw_data.columns if '_RANK' in col]
        exclude_cols.extend(rank_cols)
        
        # 选择数值特征
        feature_cols = [col for col in self.raw_data.columns 
                       if col not in exclude_cols and self.raw_data[col].dtype in ['int64', 'float64']]
        
        self.feature_names = feature_cols
        print(f"✓ 选择特征数量: {len(feature_cols)}")
        print(f"  特征列表: {feature_cols[:10]}...")
        
        return self
    
    def create_sequences(self, test_year: str = '2023-24'):
        print(f"\n创建比赛级滑动窗口序列 (每场比赛包含主队+客队, 每队窗口大小={self.window_size})...")

        df = self.raw_data.copy()

        # 编码每行(从该球队视角)是否获胜
        df['label_team'] = (df['WL'] == 'W').astype(int)

        # 确保按时间排序
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
        else:
            df = df.sort_values(['TEAM_ABBREVIATION', 'SEASON_YEAR'])

        # 为每支球队构建自身历史窗口, 然后按 GAME_ID 组合成(主队, 客队)样本
        game_dict = {}
        teams = df['TEAM_ABBREVIATION'].unique()

        for team in teams:
            team_data = df[df['TEAM_ABBREVIATION'] == team].copy()
            team_data = team_data.sort_values('GAME_DATE')

            features = team_data[self.feature_names].fillna(0).values
            labels = team_data['label_team'].values
            game_ids = team_data['GAME_ID'].values
            seasons = team_data['SEASON_YEAR'].values
            matchups = team_data['MATCHUP'].values
            dates = team_data['GAME_DATE'].values

            # 对该球队的每一场比赛(从有足够历史开始)生成一段自身历史序列
            for idx in range(self.window_size, len(team_data)):
                seq_self = features[idx-self.window_size:idx]  # 仅包含该球队的最近N场
                game_id = game_ids[idx]
                season = seasons[idx]
                matchup = matchups[idx]
                date = dates[idx]
                label_team = labels[idx]

                # 判断主客场: MATCHUP 通常形如 "LAL vs. GSW" 或 "LAL @ GSW"
                if isinstance(matchup, str) and 'vs.' in matchup:
                    side = 'home'
                elif isinstance(matchup, str) and '@' in matchup:
                    side = 'away'
                else:
                    # 无法判断主客场的比赛跳过
                    continue

                if game_id not in game_dict:
                    game_dict[game_id] = {
                        'season': season,
                        'date': date,
                        'home': None,
                        'away': None,
                    }

                game_dict[game_id][side] = {
                    'team_abbr': team,
                    'seq': seq_self,
                    'label_team': int(label_team),
                }

        train_sequences = []
        train_labels = []
        test_sequences = []
        test_labels = []

        # 遍历每一场比赛, 仅保留主客队都有足够历史的样本
        for game_id, info in game_dict.items():
            home = info.get('home')
            away = info.get('away')
            season = info.get('season')

            # 必须主客队都存在且都有历史序列
            if home is None or away is None:
                continue

            # 标签: 从主队视角, 主队赢记为1
            label = home['label_team']

            # 组合序列: 先拼接主队历史, 再拼接客队历史
            # 形状: (2 * window_size, feature_dim)
            seq_home = home['seq']
            seq_away = away['seq']

            if seq_home.shape != seq_away.shape:
                # 理论上不应该发生, 为安全起见跳过
                continue

            combined_seq = np.concatenate([seq_home, seq_away], axis=0)

            if season != test_year:
                train_sequences.append(combined_seq)
                train_labels.append(label)
            else:
                test_sequences.append(combined_seq)
                test_labels.append(label)

        train_sequences = np.array(train_sequences)
        train_labels = np.array(train_labels)
        test_sequences = np.array(test_sequences)
        test_labels = np.array(test_labels)

        print(f"✓ 训练集: {len(train_sequences)} 场比赛样本")
        print(f"✓ 测试集: {len(test_sequences)} 场比赛样本")
        if len(train_sequences) > 0:
            print(f"  训练序列形状: {train_sequences.shape} (seq_len=2*window_size, feature_dim={train_sequences.shape[-1]})")
        if len(test_sequences) > 0:
            print(f"  测试序列形状: {test_sequences.shape}")
        if len(train_labels) > 0:
            print(f"  训练集主队胜率: {train_labels.mean():.2%}")
        if len(test_labels) > 0:
            print(f"  测试集主队胜率: {test_labels.mean():.2%}")

        # 标准化特征: 将序列展平后按特征维度标准化
        if len(train_sequences) == 0 or len(test_sequences) == 0:
            raise ValueError("训练或测试样本为空，请检查数据和窗口大小设置")

        n_train, seq_len, n_features = train_sequences.shape
        train_flat = train_sequences.reshape(-1, n_features)

        self.scaler.fit(train_flat)
        train_flat_scaled = self.scaler.transform(train_flat)
        train_sequences = train_flat_scaled.reshape(n_train, seq_len, n_features)

        n_test = len(test_sequences)
        test_flat = test_sequences.reshape(-1, n_features)
        test_flat_scaled = self.scaler.transform(test_flat)
        test_sequences = test_flat_scaled.reshape(n_test, seq_len, n_features)

        self.train_sequences = train_sequences
        self.train_labels = train_labels
        self.test_sequences = test_sequences
        self.test_labels = test_labels

        return self
    
    def build_model(self, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        print("\n构建Transformer模型...")
        
        feature_dim = len(self.feature_names)
        
        self.model = TransformerPredictor(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("✓ 模型构建完成")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        return self
    
    def train(self, epochs=50, batch_size=64, lr=0.0001, weight_decay=1e-4, label_smoothing=0.1):
        print("\n" + "="*60)
        print("开始训练模型")
        print("="*60)
        
        # 创建数据加载器
        train_dataset = NBAGameDataset(self.train_sequences, self.train_labels)
        test_dataset = NBAGameDataset(self.test_sequences, self.test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 计算类别权重以处理不平衡
        unique, counts = np.unique(self.train_labels, return_counts=True)
        class_weights = torch.FloatTensor([counts.sum() / (len(unique) * c) for c in counts]).to(self.device)
        print(f"\n类别分布: {dict(zip(unique, counts))}")
        print(f"类别权重: {class_weights.cpu().numpy()}")
        
        # 损失函数和优化器 - 添加标签平滑和类别权重
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        max_patience = 15
        history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # 测试阶段
            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            
            with torch.no_grad():
                for sequences, labels in test_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_loss /= len(test_loader)
            test_acc = test_correct / test_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['lr'].append(current_lr)
            
            # 学习率调度
            scheduler.step()
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
            
            # 打印进度 - 每轮都打印以便监控
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] "
                      f"TrainLoss: {train_loss:.4f} TestLoss: {test_loss:.4f} | "
                      f"TrainAcc: {train_acc:.4f} TestAcc: {test_acc:.4f} | "
                      f"Best: {best_acc:.4f} (E{best_epoch}) | "
                      f"LR: {current_lr:.6f}")
            
            # 早停
            if patience_counter >= max_patience:
                print(f"\n早停触发! {max_patience}轮没有改善")
                break
        
        print("\n" + "="*60)
        print(f"训练完成! 最佳测试准确率: {best_acc:.4f} (Epoch {best_epoch})")
        print("="*60)
        
        self.history = history
        return best_acc
    
    def evaluate(self):
        print("\n" + "="*60)
        print("模型评估")
        print("="*60)
        
        self.model.eval()
        test_dataset = NBAGameDataset(self.test_sequences, self.test_labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"\n测试集准确率: {acc:.4f}")
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, target_names=['Loss', 'Win']))
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        return {
            'accuracy': acc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def save_model(self):
        # 保存PyTorch模型
        model_path = os.path.join(self.model_dir, "transformer_model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # 保存scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存配置
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'feature_names': self.feature_names,
                'window_size': self.window_size,
                'feature_dim': len(self.feature_names)
            }, f, ensure_ascii=False, indent=2)
    
    def load_model(self, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        config_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "transformer_model.pth")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return False
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.feature_names = config['feature_names']
            self.window_size = config['window_size']
            feature_dim = config['feature_dim']
        
        # 构建模型
        self.model = TransformerPredictor(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 加载scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("✓ 模型加载成功!")
        return True
    
    def predict_match(self, team1_history, team2_history, return_attention=False):
        self.model.eval()
        
        # 标准化
        team1_scaled = self.scaler.transform(team1_history.reshape(-1, len(self.feature_names)))
        team1_scaled = team1_scaled.reshape(1, self.window_size, -1)
        
        team2_scaled = self.scaler.transform(team2_history.reshape(-1, len(self.feature_names)))
        team2_scaled = team2_scaled.reshape(1, self.window_size, -1)
        
        # 预测
        with torch.no_grad():
            team1_tensor = torch.FloatTensor(team1_scaled).to(self.device)
            team2_tensor = torch.FloatTensor(team2_scaled).to(self.device)
            
            if return_attention:
                output1, attn1 = self.model(team1_tensor, return_attention=True)
                output2, attn2 = self.model(team2_tensor, return_attention=True)
            else:
                output1 = self.model(team1_tensor)
                output2 = self.model(team2_tensor)
                attn1, attn2 = None, None
            
            prob1 = torch.softmax(output1, dim=1)[0]  # [loss_prob, win_prob]
            prob2 = torch.softmax(output2, dim=1)[0]
            
            team1_win_prob = prob1[1].item()
            team2_win_prob = prob2[1].item()
        
        # 归一化概率
        total = team1_win_prob + team2_win_prob
        team1_win_prob_norm = team1_win_prob / total
        team2_win_prob_norm = team2_win_prob / total
        
        result = {
            'team1_win_prob': team1_win_prob_norm,
            'team2_win_prob': team2_win_prob_norm,
            'predicted_winner': 'team1' if team1_win_prob_norm > team2_win_prob_norm else 'team2',
            'confidence': abs(team1_win_prob_norm - team2_win_prob_norm)
        }
        
        if return_attention:
            result['team1_attention'] = attn1
            result['team2_attention'] = attn2
        
        return result
    
    def predict_by_team_names(self, team1_name: str, team2_name: str, save_attention=True, output_dir='prediction_output'):
        if not hasattr(self, 'raw_data') or self.raw_data is None:
            raise ValueError("请先加载数据: predictor.load_data()")
        
        # 查找球队
        team1_abbr = self._find_team(team1_name)
        team2_abbr = self._find_team(team2_name)
        
        if team1_abbr is None:
            raise ValueError(f"找不到球队: {team1_name}")
        if team2_abbr is None:
            raise ValueError(f"找不到球队: {team2_name}")
        
        # 获取最近的比赛数据
        team1_history = self._get_recent_games(team1_abbr, self.window_size)
        team2_history = self._get_recent_games(team2_abbr, self.window_size)
        
        if team1_history is None:
            raise ValueError(f"{team1_abbr} 没有足够的历史数据")
        if team2_history is None:
            raise ValueError(f"{team2_abbr} 没有足够的历史数据")
        
        # 预测
        result = self.predict_match(team1_history, team2_history, return_attention=save_attention)
        result['team1_name'] = team1_abbr
        result['team2_name'] = team2_abbr
        
        # 保存注意力热力图
        if save_attention and 'team1_attention' in result:
            os.makedirs(output_dir, exist_ok=True)
            self._visualize_attention(
                result['team1_attention'], 
                team1_abbr, 
                os.path.join(output_dir, f'attention_{team1_abbr}.png')
            )
            self._visualize_attention(
                result['team2_attention'], 
                team2_abbr, 
                os.path.join(output_dir, f'attention_{team2_abbr}.png')
            )
            print(f"\n✅ 注意力热力图已保存到: {output_dir}/")
        
        return result
    
    def _find_team(self, team_name: str):
        team_name_upper = team_name.upper()
        
        # 直接匹配缩写
        if team_name_upper in self.raw_data['TEAM_ABBREVIATION'].values:
            return team_name_upper
        
        # 匹配全名
        for abbr in self.raw_data['TEAM_ABBREVIATION'].unique():
            team_full = self.raw_data[self.raw_data['TEAM_ABBREVIATION'] == abbr]['TEAM_NAME'].iloc[0]
            if team_name.upper() in team_full.upper():
                return abbr
        
        return None
    
    def _get_recent_games(self, team_abbr: str, n_games: int):
        team_data = self.raw_data[self.raw_data['TEAM_ABBREVIATION'] == team_abbr].copy()
        team_data = team_data.sort_values('GAME_DATE', ascending=False)
        
        if len(team_data) < n_games:
            return None
        
        recent_games = team_data.head(n_games)
        recent_games = recent_games.sort_values('GAME_DATE', ascending=True)
        
        # 提取特征
        features = recent_games[self.feature_names].fillna(0).values
        return features
    
    def _visualize_attention(self, attention_weights, team_name: str, save_path: str):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
        
        if num_layers == 1:
            axes = [axes]
        
        for i, attn in enumerate(attention_weights):
            # attn shape: (batch=1, seq_len, seq_len)
            attn_matrix = attn[0].cpu().numpy()
            
            ax = axes[i]
            sns.heatmap(attn_matrix, annot=True, fmt='.2f', cmap='YlOrRd', 
                       ax=ax, cbar=True, square=True,
                       xticklabels=[f'Game {j+1}' for j in range(attn_matrix.shape[1])],
                       yticklabels=[f'Game {j+1}' for j in range(attn_matrix.shape[0])])
            ax.set_title(f'Layer {i+1} Attention', fontsize=12, fontweight='bold')
            ax.set_xlabel('Key (Past Games)', fontsize=10)
            ax.set_ylabel('Query (Past Games)', fontsize=10)
        
        plt.suptitle(f'{team_name} - Attention Heatmap', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {team_name} 注意力热力图: {save_path}")


def train_and_save():
    predictor = NBATransformerPredictor(window_size=3)
    
    # 加载和准备数据
    predictor.load_data()
    predictor.prepare_features()
    predictor.create_sequences(test_year='2023-24')
    
    # 构建和训练模型
    predictor.build_model(d_model=128, nhead=8, num_layers=4, dropout=0.1)
    predictor.train(epochs=50, batch_size=64, lr=0.001)
    
    # 评估
    predictor.evaluate()
    
    return predictor


def load_and_evaluate():
    predictor = NBATransformerPredictor(window_size=3)
    
    if not predictor.load_model():
        print("模型不存在，开始训练...")
        predictor = train_and_save()
    else:
        # 加载数据用于评估
        predictor.load_data()
        predictor.prepare_features()
        predictor.create_sequences(test_year='2023-24')
        predictor.evaluate()
    
    return predictor


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # 训练模式
        print("开始训练新模型...")
        predictor = train_and_save()
    else:
        # 评估模式
        print("加载并评估模型...")
        predictor = load_and_evaluate()
