import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F

import config
# 核心改动 1：导入我们的终极神装 DualBranchAttentionNet
from models import DualBranchAttentionNet
from load_data import get_unified_dataloaders
from utils import plot_training_curves

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=config.GAMMA, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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

    # ==========================================
    # 2. 组装模型与核武器
    # ==========================================
    # 核心改动 3：换装 DualBranchAttentionNet！
    model = DualBranchAttentionNet(
        num_channels=config.NUM_CHANNELS, 
        num_classes=config.NUM_CLASSES,
        dwt_feature_dim=378  # 18 通道 * 21 个特征
    ).to(device)
    
    criterion = FocalLoss(alpha=config.LOSS_WEIGHTS, gamma=2.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.PATIENCE, min_lr=1e-6)

    # ==========================================
    # 3. 灾备系统初始化
    # ==========================================
    os.makedirs(os.path.join(config.BASE_DIR, 'outputs', 'models'), exist_ok=True)
    best_model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', f'best_model_{test_patient}.pth')
    checkpoint_path = os.path.join(config.BASE_DIR, 'outputs', 'models', f'checkpoint_{test_patient}.pth')
    
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
        
        # 核心改动 4：接收 DataLoader 吐出的三件套！
        for batch_idx, (inputs_wave, inputs_dwt, labels) in enumerate(train_loader):
            inputs_wave = inputs_wave.to(device)
            inputs_dwt = inputs_dwt.to(device)
            labels = labels.to(device)

            use_mixup = getattr(config, 'USE_MIXUP', False)
            if use_mixup:
                mixup_alpha = getattr(config, 'MIXUP_ALPHA', 0.2)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = inputs_wave.size(0)
                index = torch.randperm(batch_size).to(device)
                
                # 双燃料混合：波形和频域特征都要混！
                inputs_wave = lam * inputs_wave + (1 - lam) * inputs_wave[index]
                inputs_dwt = lam * inputs_dwt + (1 - lam) * inputs_dwt[index]
                labels_a, labels_b = labels, labels[index]
            
            optimizer.zero_grad()
            
            # 🌟 核心改动 5：双口喂食！并且接住两个返回值！
            outputs, attn_weights = model(inputs_wave, inputs_dwt)

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
        val_preds, val_trues = [], []
        val_running_loss = 0.0
        
        print(f"Epoch {epoch+1} 结束，正在进行验证集全量体检...")
        with torch.no_grad(): 
            # 核心改动 6：验证集也要接住三件套
            for inputs_wave, inputs_dwt, labels in val_loader:
                inputs_wave = inputs_wave.to(device)
                inputs_dwt = inputs_dwt.to(device)
                labels = labels.to(device)
                
                outputs, attn_weights = model(inputs_wave, inputs_dwt)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                probs = torch.softmax(outputs.data, dim=1) 
                predicted = (probs[:, 1] > config.PREDICT_THRESHOLD).int()

                val_preds.extend(predicted.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())
                
        avg_val_loss = val_running_loss / len(val_loader)
        val_f1 = f1_score(val_trues, val_preds, pos_label=1, average='binary', zero_division=0)
        
        print("-" * 50)
        print(f"   Epoch [{epoch+1}/{config.EPOCHS}] 战报:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Val F1-Score: {val_f1:.4f}")
        
        # ----------------------------------
        # [阶段 C]：老中医号脉与防灾备份
        # ----------------------------------
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        val_f1_hist.append(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)       
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

        if val_f1 > best_val_f1:
            print(f"破纪录了！Val F1 从 {best_val_f1:.4f} 提升至 {val_f1:.4f}，正在保存神级权重...")
            best_val_f1 = val_f1
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