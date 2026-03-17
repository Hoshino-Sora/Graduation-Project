# 工具箱：utils.py
import matplotlib.pyplot as plt
import os
import config

def plot_learning_curve(loss_history, acc_history, save_path=None):
    """
    绘制双轴学习曲线，自动保存为高清图片
    """
    # 如果没传路径，就去 config 里拿默认的
    if save_path is None:
        save_path = os.path.join(config.FIG_PATH, 'training_curve.png')
        
    # 自动剥离出文件夹路径，如果文件夹不存在，自动帮你新建！
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"正在绘制训练曲线并保存至 {save_path}...")
    epochs = range(1, len(loss_history) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(epochs, loss_history, color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)  
    ax2.plot(epochs, acc_history, color=color, marker='s', label='Training Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('TCN-BiLSTM Training Learning Curve', fontsize=14, fontweight='bold')
    fig.tight_layout()  
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("曲线绘制完成！请在目录中查看。")