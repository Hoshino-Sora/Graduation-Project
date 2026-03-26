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

class DynamicWeightedCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        # 极度不平衡杀手：动态推算的正类惩罚权重
        self.weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)

    def forward(self, inputs, targets):
        self.weights = self.weights.to(inputs.device)
        return nn.functional.cross_entropy(inputs, targets, weight=self.weights)

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
    dynamic_pos_weight = (neg_samples / pos_samples) if pos_samples > 0 else 50.0
    print(f"总样本: {total_samples} | 发作: {pos_samples} | 动态暴走惩罚倍率: {dynamic_pos_weight:.2f}")

    # 2. 挂载装甲与弹药
    if config.USE_DUAL_BRANCH:
        model = DualBranchAttentionNet().to(device)
    else:
        # 为了代码统一，给纯基线包一层外壳匹配分类器接口
        model = nn.Sequential(LeftBrainTemporal(128), nn.Linear(128, 2)).to(device)

    criterion = DynamicWeightedCELoss(pos_weight=dynamic_pos_weight).to(device)
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