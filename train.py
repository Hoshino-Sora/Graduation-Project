import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import torch.nn.functional as F

import config
# 核心改动 1：导入我们的终极神装 DualBranchAttentionNet
from models import DualBranchAttentionNet, TCN_BiLSTM
from load_data import get_unified_dataloaders
from utils import plot_training_curves

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=config.GAMMA, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.reduction = reduction
#         # 将传入的 list 转换为 tensor
#         self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None

#     def forward(self, inputs, targets):
#         # 1. 计算纯净的交叉熵，绝对不能在这里加 weight！
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         # 2. 反推真实的概率 p_t
#         pt = torch.exp(-ce_loss)
#         # 3. 计算 Focal Loss 的调节因子
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
#         # 4. 最后手动乘上类别权重 (alpha)
#         if self.alpha is not None:
#             self.alpha = self.alpha.to(inputs.device)
#             alpha_t = self.alpha[targets]
#             focal_loss = alpha_t * focal_loss
            
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         else:
#             return focal_loss.sum()

class PureWeightedCELoss(nn.Module):
    def __init__(self, alpha=None, reduction='mean'):
        super(PureWeightedCELoss, self).__init__()
        self.reduction = reduction
        # 接收 config 传来的 [1.0, 50.0]
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            
        # 只要你没学好，50倍的梯度大棒就狠狠地砸下去！
        return F.cross_entropy(inputs, targets, weight=self.alpha, reduction=self.reduction)
        
def train_model(test_patient, train_patients):
    print(f"\n" + "="*50)
    print(f"正在启动 【双分支老中医】 炼丹炉 (当前测试靶标: {test_patient})")
    print("="*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"核心算力引擎: {device}")

    # ==========================================
    # 1. 呼叫后勤部：挂载全量沙箱数据
    # ==========================================
    # 核心改动 2：确保 DataLoader 吐出三件套 (默认 extract_dwt=True)
    train_loader, val_loader = get_unified_dataloaders(
        patients_list=train_patients, 
        val_ratio=0.2, 
        batch_size=config.BATCH_SIZE,
        force_positive_val=config.FORCE_POSITIVE_VAL,
        is_test=False,
        extract_dwt=config.EXTRACT_DWT  # 明确开启老中医特征提取
    )

    print("正在扫描训练集，计算极其精确的对抗权重...")
    total_samples = 0
    pos_samples = 0
    # 极速扫描一遍 DataLoader 的标签
    for batch in train_loader:
        labels = batch[-1] # 取出最后一项 (labels)
        total_samples += len(labels)
        pos_samples += labels.sum().item()
        
    neg_samples = total_samples - pos_samples
    
    if pos_samples > 0:
        # 核心物理公式：权重 = 负样本数 / 正样本数
        # 如果是 chb16，算出来的 dynamic_pos_weight 大概是 750.0！
        dynamic_pos_weight = neg_samples / pos_samples 
    else:
        dynamic_pos_weight = 1.0 # 防止除零异常

    print(f"扫描完毕！总样本: {total_samples}, 发作样本: {pos_samples}")
    print(f"动态计算得出的极其暴力的正类权重: {dynamic_pos_weight:.2f}")

    # 将算出来的恐怖权重塞进 Loss！
    dynamic_weights = [1.0, dynamic_pos_weight]

    # ==========================================
    # 2. 组装模型与核武器
    # ==========================================
    # 核心改动 3：换装 DualBranchAttentionNet！
    if config.USE_DUAL_BRANCH:
        print("启用创新架构：双分支注意力融合网络 (DualBranchAttentionNet)")
        model = DualBranchAttentionNet(
            num_channels=config.NUM_CHANNELS, 
            num_classes=config.NUM_CLASSES,
            dwt_feature_dim=378
        ).to(device)
    else:
        print("启用基线架构：纯时空黑盒 (TCN-BiLSTM)")
        model = TCN_BiLSTM(
            num_channels=config.NUM_CHANNELS, 
            num_classes=config.NUM_CLASSES
        ).to(device)
    
    criterion = PureWeightedCELoss(alpha=dynamic_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.PATIENCE, min_lr=1e-6)

    # ==========================================
    # 3. 灾备系统初始化
    # ==========================================
    os.makedirs(os.path.join(config.BASE_DIR, 'outputs', 'models'), exist_ok=True)
    model_type_str = "dual" if config.USE_DUAL_BRANCH else "baseline"
    best_model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', f'best_model_{model_type_str}_{test_patient}.pth')
    checkpoint_path = os.path.join(config.BASE_DIR, 'outputs', 'models', f'checkpoint_{model_type_str}_{test_patient}.pth')
    
    start_epoch = 0
    best_val_f1 = -1.0
    early_stop_counter = 0  
    train_loss_hist, val_loss_hist, val_f1_hist = [], [], []

    if config.RESUME_TRAINING and os.path.exists(checkpoint_path):
        print(f"\n[续训开关已开启] 检测到存档！正在恢复现场: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_f1']
        early_stop_counter = checkpoint.get('early_stop_counter', 0) 
        train_loss_hist = checkpoint.get('train_loss_hist', [])
        val_loss_hist = checkpoint.get('val_loss_hist', [])
        val_f1_hist = checkpoint.get('val_f1_hist', [])
        print(f"恢复成功！直接从第 {start_epoch + 1} 个 Epoch 继续炼丹！\n")
    else:
        print("\n[全新启动] 开启全新双分支训练流水线...\n")

    # ==========================================
    # 4. 开启史诗级训练循环
    # ==========================================
    for epoch in range(start_epoch, config.EPOCHS):
        # ----------------------------------
        # [阶段 A]：残酷的训练场
        # ----------------------------------
        model.train() 
        running_loss = 0.0
        total_batches = len(train_loader)
        
        # 核心改动 4：动态接收 DataLoader 吐出的数据
        for batch_idx, batch in enumerate(train_loader):
            if config.USE_DUAL_BRANCH:
                inputs_wave, inputs_dwt, labels = batch
                inputs_dwt = inputs_dwt.to(device)
            else:
                inputs_wave, labels = batch # 基线只有两件套
            
            inputs_wave = inputs_wave.to(device)
            labels = labels.to(device)

            use_mixup = getattr(config, 'USE_MIXUP', False)
            if use_mixup:
                mixup_alpha = getattr(config, 'MIXUP_ALPHA', 0.2)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = inputs_wave.size(0)
                index = torch.randperm(batch_size).to(device)
                
                inputs_wave = lam * inputs_wave + (1 - lam) * inputs_wave[index]
                if config.USE_DUAL_BRANCH: # Mixup也要分情况！
                    inputs_dwt = lam * inputs_dwt + (1 - lam) * inputs_dwt[index]
                labels_a, labels_b = labels, labels[index]
            
            optimizer.zero_grad()
            
            # 核心改动 5：根据架构双口喂食
            if config.USE_DUAL_BRANCH:
                outputs, attn_weights = model(inputs_wave, inputs_dwt)
            else:
                outputs = model(inputs_wave)  # 基线只吃波形

            if use_mixup:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   [Epoch {epoch+1}/{config.EPOCHS}] Batch [{batch_idx+1}/{total_batches}] | Train Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------------
        # [阶段 B]：神圣的期末考
        # ----------------------------------
        model.eval() 
        val_probs, val_trues = [], []
        val_running_loss = 0.0
        
        print(f"Epoch {epoch+1} 结束，正在进行验证集全量体检...")
        with torch.no_grad(): 
            # 核心改动 6：验证集动态解包
            for batch in val_loader:
                if config.USE_DUAL_BRANCH:
                    inputs_wave, inputs_dwt, labels = batch
                    inputs_wave = inputs_wave.to(device)
                    inputs_dwt = inputs_dwt.to(device)
                    labels = labels.to(device)
                    outputs, attn_weights = model(inputs_wave, inputs_dwt)
                else:
                    inputs_wave, labels = batch
                    inputs_wave = inputs_wave.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs_wave)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                probs = torch.softmax(outputs.data, dim=1) 
                
                # 不切断！直接收集原始概率分！
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_trues.extend(labels.cpu().numpy())
                
        avg_val_loss = val_running_loss / len(val_loader)
        # 使用 AUPRC (Average Precision) 评估排序能力！
        # 如果想看 AUROC，也可以用 val_score = roc_auc_score(val_trues, val_probs)
        val_score = average_precision_score(val_trues, val_probs)
        
        print("-" * 50)
        print(f"   Epoch [{epoch+1}/{config.EPOCHS}] 战报:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Val AUPRC Score: {val_score:.4f}")
        
        # ----------------------------------
        # [阶段 C]：老中医号脉与防灾备份
        # ----------------------------------
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        val_f1_hist.append(val_score)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_score)       
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < current_lr:
            print(f"触发降挡！模型陷入瓶颈，学习率由 {current_lr:.6f} 下调至 {new_lr:.6f}！")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_val_f1,
            'early_stop_counter': early_stop_counter,
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist,
            'val_f1_hist': val_f1_hist
        }, checkpoint_path)

        if val_score > best_val_f1:
            print(f"破纪录了！Val AUPRC 从 {best_val_f1:.4f} 提升至 {val_score:.4f}，正在保存排序之王权重...")
            best_val_f1 = val_score
            early_stop_counter = 0  
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            print(f"早停警告: 连续 {early_stop_counter}/{config.EARLY_STOP_PATIENCE} 轮无提升。")
            if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n触发早停机制！模型已陷入过拟合，切断训练！")
                break
                
        print("-" * 50 + "\n")

    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    print(f"\n训练全部结束！最高 Val F1 分数锁定在: {best_val_f1:.4f}")

# 本地测试入口已注释，实际通过 run_pipeline.py 调用
# if __name__ == "__main__":
#     train_model("chb01", ["chb02", "chb03"])