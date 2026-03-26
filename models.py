# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from preprocessing import RobustZScoreNorm, RelativePowerNorm

# ==========================================
# 左脑：TCN-BiLSTM (时序波形专家)
# ==========================================
class LeftBrainTemporal(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.norm = RobustZScoreNorm()
        self.conv = nn.Sequential(
            nn.Conv1d(config.NUM_CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=config.LSTM_HIDDEN_SIZE, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.LSTM_HIDDEN_SIZE * 2, out_dim)

    def forward(self, x_wave):
        x_wave = self.norm(x_wave)          # TTA 物理清洗
        h = self.conv(x_wave)               # [B, 32, L]
        h = h.permute(0, 2, 1)              # [B, L, 32]
        _, (h_n, _) = self.lstm(h)          # 取最后时刻状态
        h_fused = torch.cat((h_n[0], h_n[1]), dim=-1) # [B, 128]
        out = self.fc(h_fused)
        return torch.tanh(out)              # 锁死 [-1, 1] 防止振幅霸权抢麦

# ==========================================
# 右脑：Channel-Aware 频域神探 (原生可解释性)
# ==========================================
class RightBrainFrequency(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.relative_norm = RelativePowerNorm()
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.DWT_FEATURE_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )
        if config.USE_CHANNEL_ATTENTION:
            self.attention_scorer = nn.Sequential(
                nn.Linear(out_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1)            # 为18个通道各自打分
            )

    def forward(self, x_dwt):
        # x_dwt 需预先 reshape 为 [B, 18, 21]
        batch_size = x_dwt.size(0)
        x_dwt = x_dwt.view(batch_size, config.NUM_CHANNELS, config.DWT_FEATURE_DIM)
        x_dwt = self.relative_norm(x_dwt)   # 物理清洗：只看相对旋律
        
        h_channels = self.feature_extractor(x_dwt) # [B, 18, 128]
        
        if config.USE_CHANNEL_ATTENTION:
            scores = self.attention_scorer(h_channels) # [B, 18, 1]
            attn_weights = F.softmax(scores, dim=1)    # 极其重要的可解释性地形图！
            h_fused = torch.sum(h_channels * attn_weights, dim=1) # [B, 128]
        else:
            h_fused = torch.mean(h_channels, dim=1)
            attn_weights = None
            
        return torch.tanh(h_fused), attn_weights # 同样锁死 [-1, 1]

# ==========================================
# 中央审判庭：双重注意力博弈网络 (Double Attention)
# ==========================================
class DualBranchAttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_brain = LeftBrainTemporal(out_dim=128)
        self.right_brain = RightBrainFrequency(out_dim=128)
        
        # 模态注意力打分器 (评估到底该听谁的)
        self.modality_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        
        # 修正：输入依然是 256 维！我们绝不压缩特征，我们只调节它们的“音量”！
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(64, 2)
        )

    def forward(self, x_wave, x_dwt):
        # 1. 提取单边特征
        h_left = self.left_brain(x_wave)
        h_right, channel_attn = self.right_brain(x_dwt) 
        
        # 2. 模态注意力博弈 (Modality Attention)
        concat_h = torch.cat((h_left, h_right), dim=1) # [B, 256]
        
        # 计算左右脑各自的权重 (Softmax 保证两者之和为 1)
        modality_scores = self.modality_attention(concat_h)
        modality_weights = F.softmax(modality_scores, dim=1) # [B, 2]
        
        alpha_left = modality_weights[:, 0].unsqueeze(1)  # [B, 1]
        alpha_right = modality_weights[:, 1].unsqueeze(1) # [B, 1]
        
        # 3. 终极融合：带门控的拼接 (Gated Concatenation)
        # 用权重去放大或缩小各自的特征，但保持它们 128 维的独立性！
        h_left_weighted = h_left * alpha_left
        h_right_weighted = h_right * alpha_right
        
        # 拼接后的 256 维，前 128 维是调过音量的左脑，后 128 维是调过音量的右脑
        h_fused = torch.cat((h_left_weighted, h_right_weighted), dim=1) 
        
        # 4. 最终裁决
        logits = self.classifier(h_fused)
        
        # 返回分类结果，以及极其豪华的【两套可解释性权重】！
        return logits, (channel_attn, modality_weights)