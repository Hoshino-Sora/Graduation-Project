# config.py
import os
import sys

# ==========================================
# 1. 全局路径与环境配置
# ==========================================
if sys.platform.startswith('linux') and os.path.exists('/root/autodl-tmp'):
    BASE_DIR = '/root/autodl-tmp/Graduation_Project'
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.path.join(BASE_DIR, 'datasets')
PROCESSED_DATA_PATH = os.path.join(DATASET_ROOT, 'processed_chbmit')
FIG_PATH = os.path.join(BASE_DIR, 'outputs', 'figures')
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'models')

for p in [PROCESSED_DATA_PATH, FIG_PATH, MODEL_PATH]:
    os.makedirs(p, exist_ok=True)

# ==========================================
# 2. 物理与信号超参数
# ==========================================
CHBMIT_FS = 256
CHBMIT_WINDOW_SEC = 2
CHBMIT_TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
    'FZ-CZ', 'CZ-PZ'
]
NUM_CHANNELS = len(CHBMIT_TARGET_CHANNELS)  # 18
DWT_FEATURE_DIM = 21  # 老中医每通道提取的物理特征数
NUM_CLASSES = 2       # 0:正常, 1:发作

# ==========================================
# 3. 终极架构消融实验开关 (Ablation Switches)
# ==========================================
# --- 架构级 ---
USE_DUAL_BRANCH = True           # True: 双脑神装 | False: 退化为纯 TCN-BiLSTM 基线
EXTRACT_DWT = USE_DUAL_BRANCH    # 联动开关
# --- 特征清洗级 (TTA) ---
USE_INDEPENDENT_Z_SCORE = True   # 【护城河】是否开启通道独立 Z-Score 防爆 (抗绝对底噪)
USE_RELATIVE_POWER_L2 = True     # 【护城河】是否开启频段 L2 归一化 (抗绝对振幅)
# --- 可解释性级 ---
USE_CHANNEL_ATTENTION = True     # 右脑是否使用 Channel Attention (False 则退化为 Mean Pooling)

# ==========================================
# 4. 炼丹炉控制系统
# ==========================================
BATCH_SIZE = 1024            # 内存允许的情况下尽量大，稳定梯度
LEARNING_RATE = 1e-4        # 在大权重下，学习率必须求稳
EPOCHS = 100
RANDOM_SEED = 42

LSTM_HIDDEN_SIZE = 64
DROPOUT_RATE = 0.3
GRADIENT_CLIP_NORM = 1.0    # 关键：防暴走权重导致 NaN 的绝对保险丝

# 调度器与早停
PATIENCE = 6                # LR 降档耐心值
EARLY_STOP_PATIENCE = 20    # 彻底死心拔电源的耐心值
RESUME_TRAINING = False

# ==========================================
# 5. 临床评估后处理兜底机制
# ==========================================
USE_ADAPTIVE = True
TARGRT_PERCENTILE = 98      # 动态 P=98% 截断及格线
PREDICT_THRESHOLD_TEST = 0.2  # 固定阈值兜底参数
SMOOTHING_WINDOW = 5        # 滑动平滑消除孤立毛刺
COLLAR_TOLERANCE = 5.0      # 容差秒数