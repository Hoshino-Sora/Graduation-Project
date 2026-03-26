# preprocessing.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class RobustZScoreNorm(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps # 吉洪诺夫正则化防爆锁

    def forward(self, x):
        """输入: [Batch, Channels, TimeSteps]"""
        if not config.USE_INDEPENDENT_Z_SCORE:
            return x
        # 绝对通道独立 dim=-1，誓死捍卫通道间贫富差距！
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 截断死寂通道，防止热噪声被放大成核爆
        std = torch.clamp(std, min=self.eps)
        return (x - mean) / std

class RelativePowerNorm(nn.Module):
    def forward(self, x_dwt):
        """输入: [Batch, Channels, FreqFeatures]"""
        if not config.USE_RELATIVE_POWER_L2:
            return x_dwt
        # 取 log1p 抹平极端尖峰，再做 L2 归一化提取相对旋律
        x_dwt = torch.log1p(torch.abs(x_dwt))
        return F.normalize(x_dwt, p=2, dim=-1, eps=1e-8)