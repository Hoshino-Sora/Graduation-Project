import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ==========================================
# 左脑：传统黑盒时序特征提取 (复用基线)
# ==========================================
class TCN_BiLSTM(nn.Module):
    def __init__(self, num_channels=18, num_classes=2):
        """
        癫痫脑电检测的核心架构：TCN-BiLSTM
        :param num_channels: 输入的脑电通道数 (CHB-MIT是18，Bonn是1)
        :param num_classes: 分类数 (0:正常, 1:发作)
        """
        super(TCN_BiLSTM, self).__init__()
        
        # ==========================================
        # 1. TCN 模块 (时间卷积网络) - 负责捕捉局部高频突变
        # 输入维度要求: (Batch_Size, Channels, Sequence_Length)
        # ==========================================
        self.tcn = nn.Sequential(
            # 第一层卷积：提取浅层形态特征
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 下采样，长度减半
            
            # 第二层卷积：提取深层高维特征 (使用稍大的空洞率如果需要，这里用普通卷积演示)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度再次减半
        )
        
        # ==========================================
        # 2. BiLSTM 模块 (双向长短期记忆网络) - 负责结合长程上下文逻辑
        # 输入维度要求: (Batch_Size, Sequence_Length, Features)
        # ==========================================
        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.bilstm = nn.LSTM(
            input_size=64,             # 必须和 TCN 最后一层的 out_channels 一致
            hidden_size=self.lstm_hidden_size, 
            num_layers=2,              # 叠加两层 LSTM 增加拟合能力
            batch_first=True,          # 明确告诉 PyTorch，数据的第0维是 Batch_Size
            bidirectional=True         # 开启双向：同时参考过去和未来的波形
        )
        
        # ==========================================
        # 3. 全连接分类器 (Classifier) - 负责输出最终判定
        # ==========================================
        # 因为是双向 LSTM，所以隐层特征数要乘以 2
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),         # 极其重要：防止过拟合的 Dropout
            nn.Linear(32, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        前向传播逻辑，注意这里的维度变换 (Tensor Shape Shifting)
        """
        # 1. 过 TCN 提取局部特征
        # 输入 x 形状: [Batch, 18, 4097] (以原始切片为例)
        out = self.tcn(x) 
        # 输出 out 形状: [Batch, 64, 1024] (通道数变多，序列长度被 MaxPool 缩短)
        
        # 维度大挪移！
        # CNN 喜欢 [Batch, Channel, Length]
        # RNN 喜欢 [Batch, Length, Channel(Feature)]
        out = out.permute(0, 2, 1) 
        # 现在 out 形状变成了: [Batch, 1024, 64]
        
        # 2. 过 BiLSTM 提取时序逻辑
        out, (h_n, c_n) = self.bilstm(out)
        # 我们只需要序列最后一个时间步的输出，或者取隐状态 h_n
        # h_n 包含了整个序列的浓缩精华。它的形状是 [num_layers * num_directions, Batch, Hidden_Size]
        
        # 提取最后一层的前向和后向隐状态并拼接
        # forward_hidden: h_n[-2, :, :]
        # backward_hidden: h_n[-1, :, :]
        final_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) 
        # final_hidden 形状: [Batch, 128]
        
        # 3. 喂给全连接层输出概率分布 (Logits)
        # 如果是双分支架构调用，我们只返回 128 维的特征，不急着出结果！
        if return_features:
            return final_hidden
            
        logits = self.classifier(final_hidden)
        return logits

# ==========================================
# 右脑：老中医白盒频域特征提取 (MLP)
# ==========================================
class PriorFeatureBranch(nn.Module):
    def __init__(self, input_dim=378, hidden_dim=128):
        """
        专门用来消化 DWT 小波特征的多层感知机
        :param input_dim: 18个通道 * 21个特征 = 378
        :param hidden_dim: 必须和左脑输出的维度一致 (默认 128)
        """
        super(PriorFeatureBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)

# ==========================================
# 中央大脑皮层：双分支注意力融合网络 (终极神装)
# ==========================================
class DualBranchAttentionNet(nn.Module):
    def __init__(self, num_channels=18, num_classes=2, dwt_feature_dim=378):
        super(DualBranchAttentionNet, self).__init__()
        
        # 1. 挂载左脑
        self.temporal_branch = TCN_BiLSTM(num_channels=num_channels, num_classes=num_classes)
        # 提取左脑隐层维度 (LSTM_HIDDEN * 2) -> 默认是 64 * 2 = 128
        temporal_out_dim = self.temporal_branch.lstm_hidden_size * 2 
        
        # 2. 挂载右脑
        self.frequency_branch = PriorFeatureBranch(input_dim=dwt_feature_dim, hidden_dim=temporal_out_dim)
        
        # 3. 核心大招：动态注意力打分器 (Attention Scorer)
        # 它看一眼左脑和右脑的汇总报告 (128 + 128 = 256)，然后决定听谁的！
        self.attention_scorer = nn.Sequential(
            nn.Linear(temporal_out_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 2) # 输出 2 个打分 (分别对应左脑和右脑)
        )
        
        # 4. 最终裁决器
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_wave, x_dwt):
        # 1. 左脑出列：计算黑盒时序特征 [Batch, 128]
        h_t = self.temporal_branch(x_wave, return_features=True)
        
        # 2. 右脑出列：计算白盒频域特征 [Batch, 128]
        x_dwt_log = torch.log1p(torch.abs(x_dwt))

        h_f = self.frequency_branch(x_dwt_log)
        
        # 3. 皮层开会：打分器介入
        # 拼接特征供打分器审阅 [Batch, 256]
        concat_h = torch.cat((h_t, h_f), dim=1)
        
        # 算出初步打分 [Batch, 2]
        attn_logits = self.attention_scorer(concat_h)
        
        # 使用 Softmax 强制转换为百分比权重 (加起来等于 100%)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # 提取并扩充维度 [Batch, 1]
        alpha_t = attn_weights[:, 0].unsqueeze(1) # 左脑听信度
        alpha_f = attn_weights[:, 1].unsqueeze(1) # 右脑听信度
        
        # 终极操作：按权重动态融合！如果遇到咬牙伪影，alpha_f (右脑) 的权重会自动飙升！
        h_fused = alpha_t * h_t + alpha_f * h_f
        
        # 4. 最终出分 [Batch, 2]
        logits = self.classifier(h_fused)
        
        # 注意：我们不仅返回预测结果，还把注意力权重一起返回了，方便以后画 SHAP 图和注意力热力图！
        return logits, attn_weights


# --- 独立联调测试 (Sanity Check) ---
if __name__ == "__main__":
    print("=== 终极神装 DualBranchAttentionNet 维度测试 ===")
    
    batch_size = 16
    # 模拟从 dataloader 抓出来的左脑数据 (波形)
    dummy_wave = torch.randn(batch_size, 18, 512) 
    # 模拟从 dataloader 抓出来的右脑数据 (DWT物理特征)
    dummy_dwt = torch.randn(batch_size, 378)
    
    print(f"黑盒时序输入: {dummy_wave.shape}")
    print(f"白盒频域输入: {dummy_dwt.shape}")
    
    model = DualBranchAttentionNet(num_channels=18, num_classes=2, dwt_feature_dim=378)
    
    if torch.cuda.is_available():
        print("检测到 GPU，转移战场...")
        model = model.cuda()
        dummy_wave = dummy_wave.cuda()
        dummy_dwt = dummy_dwt.cuda()
        
    # 前向传播跑一次 (注意要接住两个返回值！)
    output_logits, attention_weights = model(dummy_wave, dummy_dwt)
    
    print(f"模型最终分类预测维度: {output_logits.shape} -> [Batch, Classes]")
    print(f"模型内部注意力权重维度: {attention_weights.shape} -> [Batch, 2] (对应左脑和右脑的信任度)")
    
    # 偷偷看一眼第一个样本的信任度分配
    sample_0_weights = attention_weights[0].detach().cpu().numpy()
    print(f"样本 0 的注意力分配 -> 左脑(TCN): {sample_0_weights[0]:.2f}, 右脑(DWT): {sample_0_weights[1]:.2f}")
    
    print("测试完美通过！网络不仅维度无缝衔接，还长出了具备自我调节能力的‘大脑皮层’！")