import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F

# 导入咱们自己的模块
import config
from models import TCN_BiLSTM
from load_data import get_unified_dataloaders
from utils import plot_training_curves

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # 如果传入了权重，保存下来
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # 1. 如果有 alpha，把它放到和 inputs 一样的 GPU/CPU 上
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            
        # 2. 计算基础的 CrossEntropy (但不求平均，保留每个样本的单独 loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # 3. 核心魔法：推导出 pt (模型对真实标签的预测概率)
        # 因为 ce_loss = -log(pt)，所以 pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 4. 乘上焦点调节因子 (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_model():
    print("正在启动 TCN-BiLSTM 炼丹炉...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"核心算力引擎: {device}")

    # ==========================================
    # 1. 呼叫后勤部：挂载全量沙箱数据
    # ==========================================
    # 目标：跑全集 24 个病人
    train_patients =[f"chb{i:02d}" for i in range(2, 25)]
    train_loader, val_loader = get_unified_dataloaders(
        patients_list=train_patients, 
        val_ratio=0.2, 
        batch_size=config.BATCH_SIZE,
        force_positive_val=config.FORCE_POSITIVE_VAL,
        is_test=False
    )

    # ==========================================
    # 2. 组装模型与核武器（损失函数 & 优化器）
    # ==========================================
    model = TCN_BiLSTM(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
    
    # 换上终极核武器 (结合了 89 倍惩罚和 Gamma=2 的聚焦)
    criterion = FocalLoss(alpha=config.LOSS_WEIGHTS, gamma=2.0).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # 自动变速箱：盯着 Validation F1，连续 3 轮不涨，学习率砍半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6)

    # ==========================================
    # 3. 云端防断连断点续训机制 (带物理安全锁)
    # ==========================================
    os.makedirs(os.path.join(config.BASE_DIR, 'outputs', 'models'), exist_ok=True)
    best_model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', 'best_model.pth')
    checkpoint_path = os.path.join(config.BASE_DIR, 'outputs', 'models', 'latest_checkpoint.pth')
    
    start_epoch = 0
    best_val_f1 = 0.0
    early_stop_counter = 0  # 新增：早停计数器
    train_loss_hist = []
    val_loss_hist = []
    val_f1_hist = []

    # 核心改动：必须同时满足“开关打开”且“存档存在”才读档！
    if config.RESUME_TRAINING and os.path.exists(checkpoint_path):
        print(f"\n[续训开关已开启] 检测到存档！正在恢复现场: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_f1']
        # 恢复早停计数器（如果之前存档里有的话）
        early_stop_counter = checkpoint.get('early_stop_counter', 0) 
        
        train_loss_hist = checkpoint.get('train_loss_hist', [])
        val_loss_hist = checkpoint.get('val_loss_hist', [])
        val_f1_hist = checkpoint.get('val_f1_hist', [])
        
        print(f"恢复成功！将直接从第 {start_epoch + 1} 个 Epoch 继续炼丹！历史最高 F1: {best_val_f1:.4f}\n")
    else:
        print("\n[全新启动] 断点续训未开启或无旧存档，无视干扰，开启全新训练流水线...\n")

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
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪：防止 89 倍惩罚把梯度搞爆炸
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
        val_preds = []
        val_trues =[]
        val_running_loss = 0.0
        
        print(f"Epoch {epoch+1} 结束，正在进行验证集全量体检...")
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                # --- 爆改后的代码 (0.3 敏感阈值) ---
                # 1. 先用 softmax 把模型输出变成总和为 1 的概率分布
                probs = torch.softmax(outputs.data, dim=1) 

                # 2. 强行设定：只要第 1 类（发作）的概率大于 0.3，就判为 1，否则为 0
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
        #[阶段 C]：老中医号脉与防灾备份
        # ----------------------------------
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        val_f1_hist.append(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)       
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < current_lr:
            print(f"触发自动降挡！模型陷入瓶颈，学习率由 {current_lr:.6f} 下调至 {new_lr:.6f}！")
        
        # 无条件保存最新的灾备存档点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_val_f1,
            'early_stop_counter': early_stop_counter, # 存入计数器
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist,
            'val_f1_hist': val_f1_hist
        }, checkpoint_path)
        print(f"最新训练进度已存档: {checkpoint_path}")

        # 核心改动：破纪录与早停机制的博弈
        if val_f1 > best_val_f1:
            print(f"破纪录了！Val F1 从 {best_val_f1:.4f} 提升至 {val_f1:.4f}，正在保存神级权重...")
            best_val_f1 = val_f1
            early_stop_counter = 0  # 只要破纪录，早停计数器立刻清零！
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            print(f"早停警告: 连续 {early_stop_counter}/{config.EARLY_STOP_PATIENCE} 轮未打破记录。")
            
            if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n触发早停机制！连续 {config.EARLY_STOP_PATIENCE} 轮 F1 无提升。")
                print("模型已进入严重过拟合状态，切断训练！")
                break # 提前下班！
                
        print("-" * 50 + "\n")

    print(f"训练全部结束！最高 Val F1 分数锁定在: {best_val_f1:.4f}")
    print(f"最佳权重已安放在: {best_model_path}")
    plot_training_curves(train_loss_hist, val_loss_hist, val_f1_hist)

if __name__ == "__main__":
    train_model()