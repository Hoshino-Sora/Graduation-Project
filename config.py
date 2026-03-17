import os

# ==========================================
# 1. 全局路径配置
# ==========================================
# 获取当前 config.py 所在的绝对目录 (项目根目录)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集根目录
DATASET_ROOT = os.path.join(BASE_DIR, 'datasets')

# 子数据集路径
BONN_DATA_PATH = os.path.join(DATASET_ROOT, 'bonn')
CHBMIT_DATA_PATH = os.path.join(DATASET_ROOT, 'chbmit')

# 预处理后数据的输出总目录
PROCESSED_DATA_PATH = os.path.join(DATASET_ROOT, 'processed_chbmit')

# 专门存放自动生成的各种图表的目录
FIG_PATH = os.path.join(BASE_DIR, 'outputs', 'figures')

# ==========================================
# 2. Bonn 数据集配置 (短片段单通道)
# ==========================================s
BONN_FS = 173.61                 # 采样率 (Hz)
BONN_TOTAL_POINTS = 4097         # 原始单文件数据点数
BONN_WINDOW_SIZE = 409           # 切窗大小 (409点约=2.36秒)
# 标签映射：二分类任务 (发作 vs 非发作)
BONN_FOLDER_MAPPING = {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1}

# ==========================================
# 3. CHB-MIT 数据集配置 (长程多通道)
# ==========================================
CHBMIT_FS = 256                  # 采样率 (Hz)
CHBMIT_WINDOW_SEC = 2            # 切窗时长 (秒)
CHBMIT_OVERLAP_SEC = 0           # 切窗重叠时长 (秒)

# 物理阉割：强制通道对齐 (跨患者一致性的 18 个核心双极导联通道)
CHBMIT_TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
    'FZ-CZ', 'CZ-PZ'
]

# ==========================================
# 4. 模型与训练配置 (预留，下周用)
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 1
RANDOM_SEED = 42                 # 固定随机种子，保证实验可复现

# 新增：网络架构超参数 (消融实验必备)
LSTM_HIDDEN_SIZE = 64            # BiLSTM 隐藏层维度
DROPOUT_RATE = 0.5               # 全连接层防过拟合概率

# 动态读取通道数，防止后续修改通道列表时漏改！
NUM_CHANNELS = len(CHBMIT_TARGET_CHANNELS)  # 自动算出 18
NUM_CLASSES = 2                             # 0:正常, 1:发作

# 针对 1:89 极度不平衡数据的“狙击手权重” 
# 0代表正常(权重1)，1代表发作(权重89，严惩漏报！)
LOSS_WEIGHTS = [1.0, 89.0]

# ==========================================
# 5. 临床后处理与评估参数 (Post-Processing)
# ==========================================
# 平滑滤波窗口大小 (必须是奇数)。
# 设为 5 意味着：看前后一共 10 秒的波形来决定中间这 2 秒是不是真发作
SMOOTHING_WINDOW = 5       

# Collar 容差评分区间 (单位：秒)
# 国际脑电标准：只要预测发作时间在真实发作前后的 5 秒内，就算命中！
COLLAR_TOLERANCE = 5.0