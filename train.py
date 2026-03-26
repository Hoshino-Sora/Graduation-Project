# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score
import config
from models import DualBranchAttentionNet, LeftBrainTemporal # 兼容单/双分支
from load_data import get_unified_dataloaders
from utils import plot_training_curves
import torch.nn.functional as F

class AdvancedFocalLoss(nn.Module):
    def __init__(self, alpha_weight=20.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # 将传入的动态权重转化为 tensor (0类正常为1.0, 1类发作为alpha_weight)
        self.alpha = torch.tensor([1.0, alpha_weight], dtype=torch.float32)

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)
        
        # 1. 计算标准交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. 获取模型对真实标签的预测概率 pt
        pt = torch.exp(-ce_loss)
        
        # 3. Focal Loss 核心机制：(1 - pt)^gamma
        # 预测越准 (pt接近1)，Loss 压得越低！让模型别在简单底噪上浪费精力！
        focal_term = (1 - pt) ** self.gamma
        
        # 4. 结合类别权重
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def train_model(test_patient, train_patients):
    print(f"\n{'='*50}\n启动 V2.0 超级炼丹炉 (当前靶标: {test_patient})\n{'='*50}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_unified_dataloaders(
        patients_list=train_patients, val_ratio=0.2, 
        batch_size=config.BATCH_SIZE, is_test=False, extract_dwt=config.EXTRACT_DWT
    )

    # 1. 动态核算暴走权重
    print("正在核算战场敌我兵力比...")
    # 瞬间核算战场兵力，拒绝全盘迭代！
    total_samples, pos_samples = 0, 0
    for ds in train_loader.dataset.datasets:
        total_samples += ds.num_samples
        pos_samples += np.sum(ds.y_disk)
    total_samples = len(train_loader.dataset)
    neg_samples = total_samples - pos_samples
    if pos_samples > 0:
        # 原始比例
        raw_weight = neg_samples / pos_samples 
        # 强行截断！绝不许超过 20 倍！防止网络变成被害妄想症！
        dynamic_pos_weight = min(raw_weight, 20.0) 
    else:
        dynamic_pos_weight = 1.0

    print(f"扫描完毕！总样本: {total_samples}, 发作样本: {pos_samples}")
    print(f"原始算术比例: {raw_weight:.2f} | 截断后装备的狙击权重: {dynamic_pos_weight:.2f}")

    # 2. 挂载装甲与弹药
    if config.USE_DUAL_BRANCH:
        model = DualBranchAttentionNet().to(device)
    else:
        # 为了代码统一，给纯基线包一层外壳匹配分类器接口
        model = nn.Sequential(LeftBrainTemporal(128), nn.Linear(128, 2)).to(device)

    criterion = AdvancedFocalLoss(alpha_weight=dynamic_pos_weight, gamma=2.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.PATIENCE)

    best_val_auprc = -1.0
    early_stop_counter = 0
    model_type_str = "dual" if config.USE_DUAL_BRANCH else "baseline"
    best_model_path = os.path.join(config.MODEL_PATH, f'best_model_{model_type_str}_{test_patient}.pth')

    # 3. 史诗级训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if config.USE_DUAL_BRANCH:
                inputs_wave, inputs_dwt, labels = [b.to(device) for b in batch]
                outputs, _ = model(inputs_wave, inputs_dwt)
            else:
                inputs_wave, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs_wave)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 神级保险丝：防核爆梯度裁剪！
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        # 验证期末考 (唯一真神：AUPRC)
        model.eval()
        val_probs, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                if config.USE_DUAL_BRANCH:
                    inputs_wave, inputs_dwt, labels = [b.to(device) for b in batch]
                    outputs, _ = model(inputs_wave, inputs_dwt)
                else:
                    inputs_wave, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs_wave)
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_trues.extend(labels.cpu().numpy())

        # 绝不看 Accuracy 和 F1！只看纯粹的排序能力 AUPRC！
        if sum(val_trues) == 0:
            val_auprc = 0.0 # 防空载报错锁
            print("警告：当前验证集无正样本，AUPRC 置零。")
        else:
            val_auprc = average_precision_score(val_trues, val_probs)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Val AUPRC: {val_auprc:.4f}")

        scheduler.step(val_auprc)

        if val_auprc > best_val_auprc:
            print(f"破纪录！AUPRC 从 {best_val_auprc:.4f} 飙升至 {val_auprc:.4f}，保存神装！")
            best_val_auprc = val_auprc
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                print(f"触发早停机制，模型已熬出极品原汤，切断训练！")
                break