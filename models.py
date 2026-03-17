import torch
import torch.nn as nn
import config

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

    def forward(self, x):
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
        logits = self.classifier(final_hidden)
        # logits 形状: [Batch, 2]
        
        return logits

# --- 独立联调测试 (Sanity Check) ---
if __name__ == "__main__":
    print("=== TCN-BiLSTM 架构维度测试 ===")
    
    # 假设你的 Batch Size 是 16，18 个通道，每个窗口 4097 个采样点 (约16秒的256Hz信号)
    dummy_input = torch.randn(16, 18, 4097) 
    print(f"模拟输入数据维度: {dummy_input.shape} -> [Batch, Channels, SeqLen]")
    
    # 初始化模型
    model = TCN_BiLSTM(num_channels=18, num_classes=2)
    
    # 如果有 GPU，把模型和数据都扔进炼丹炉
    if torch.cuda.is_available():
        print("检测到 GPU，正在将张量转移到显存...")
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        
    # 前向传播跑一次
    output = model(dummy_input)
    
    print(f"模型输出结果维度: {output.shape} -> [Batch, Classes]")
    print("测试通过！这套网络没有维度冲突，可以完美串联！")