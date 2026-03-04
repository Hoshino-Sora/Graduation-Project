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
LEARNING_RATE = 1e-3
EPOCHS = 50
RANDOM_SEED = 42                 # 固定随机种子，保证实验可复现