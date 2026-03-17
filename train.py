import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils import plot_learning_curve

# 导入模组
import config
from models import TCN_BiLSTM
from load_data import get_patient_dataloader

def train_model():
    print("=== TCN-BiLSTM 炼丹炉正式点火 ===")
    
    # ==========================================
    # 1. 硬件探测与超参数设置 (Hyperparameters)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运算引擎: {device} (请确认是 cuda!)")
    
    EPOCHS = config.EPOCHS                  # 训练轮数
    BATCH_SIZE = config.BATCH_SIZE          # 3070显存有8G，64完全吃得下
    LEARNING_RATE = config.LEARNING_RATE    # 经典初始学习率

    # ==========================================
    # 2. 挂载真实数据管道
    # ==========================================
    
    # 优雅地要到 chb01 的全部弹药！
    dataloader = get_patient_dataloader(patient_id="chb01", 
                                        batch_size=config.BATCH_SIZE, 
                                        shuffle=True)

    # ==========================================
    # 3. 初始化模型、损失函数与优化器
    # ==========================================
    model = TCN_BiLSTM(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
    
    # 损失函数：交叉熵 (CrossEntropyLoss)。它天生自带 Softmax，不需要在模型最后一层加 Softmax！
    # 把 1:89 的权重送进 Loss！
    class_weights = torch.tensor(config.LOSS_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器：Adam (自适应动量梯度下降)，深度学习万金油
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 4. 正式开启训练大循环 (The Training Loop)
    # ==========================================
    print("\n开始训练！观察 Loss 是否下降...")
    train_loss_history = []
    train_acc_history = []
    for epoch in range(EPOCHS):
        model.train() # 开启训练模式 (启用 Dropout 和 BatchNorm)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # 把张量搬运到 GPU 显存里
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 步步为营的“炼丹四步曲”：
            # Step 1: 梯度清零 (极其重要，否则梯度会无限累加)
            optimizer.zero_grad()
            
            # Step 2: 前向传播 (让网络猜一次)
            outputs = model(inputs)
            
            # Step 3: 计算误差 (算算网络猜得有多离谱)
            loss = criterion(outputs, labels)
            
            # Step 4: 反向传播与权重更新 (根据误差修正网络里的几百万个参数)
            loss.backward()
            # 梯度裁剪 (防止 89 倍权重导致的梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # --- 统计进度与准确率 ---
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # 打印每个 Batch 的进度
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"   Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(dataloader)}] | "
                      f"Loss: {loss.item():.4f}")
                
        # 计算每一轮的平均指标
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = (correct_predictions / total_samples) * 100
        print(f"[Epoch {epoch+1} 战报] 平均 Loss: {epoch_loss:.4f} | 训练准确率: {epoch_acc:.2f}%\n")
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

    print("太好了！你的 TCN-BiLSTM 已经成功完成了小批量数据的闭环训练！")
    # 呼叫画图函数！
    plot_learning_curve(train_loss_history, train_acc_history)

    # 保存最新的模型权重！
    # 把它统一存到 outputs 文件夹下，永远叫 latest_model.pth
    model_save_path = os.path.join(config.BASE_DIR, 'outputs', 'models', 'latest_model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 提取模型的“灵魂”（参数字典）并物理写入硬盘
    torch.save(model.state_dict(), model_save_path)
    print(f"炼丹结束！最新模型权重已覆盖并保存至: {model_save_path}")


if __name__ == "__main__":
    train_model()