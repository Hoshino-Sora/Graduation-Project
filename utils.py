import matplotlib.pyplot as plt
import os
import config

def plot_training_curves(train_losses, val_losses, val_f1s):
    """
    终极双联屏监控：左边看 Loss 缠斗，右边看 F1 飙车
    """
    os.makedirs(os.path.join(config.BASE_DIR, 'outputs', 'figures'), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 画一张 12x5 的宽屏大画布
    plt.figure(figsize=(12, 5))
    
    # 左图：Loss 曲线对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r--', label='Val Loss', linewidth=2)
    plt.title('Loss: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True)
    
    # 右图：F1 黄金指标趋势
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1s, 'g-o', label='Val F1-Score', linewidth=2, markersize=6)
    plt.title('Medical Gold Standard: Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(config.BASE_DIR, 'outputs', 'figures', 'training_dashboard.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n双联屏训练战报已高清保存至: {save_path}")
    plt.close()