import numpy as np
from sklearn.preprocessing import StandardScaler
import config
import mne

def preprocess_eeg(X, y, window_size=config.BONN_WINDOW_SIZE):
    """
    对脑电数据进行标准化和切窗扩增
    :param X: 原始数据，形状 (Samples, TimeSteps) -> (500, 4097)
    :param y: 原始标签，形状 (Samples,) -> (500,)
    :param window_size: 每个时间窗的长度 (409点 约等于 2.36秒)
    :return: X_processed, y_processed
    """
    # 1. Z-score 标准化 (对每个样本单独进行，消除个体电位基线差异)
    # 注意：这里不能把所有数据混在一起标准化，必须按行（样本）处理
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        scaler = StandardScaler()
        # reshape(-1, 1) 是因为 sklearn 要求二维输入
        X_norm[i] = scaler.fit_transform(X[i].reshape(-1, 1)).flatten()
        
    # 2. 切窗操作 (Window Slicing)
    X_sliced = []
    y_sliced = []
    
    # 计算每个原始样本可以切出多少个完整的窗口
    num_windows = X.shape[1] // window_size
    
    for i in range(X.shape[0]):
        for j in range(num_windows):
            start_idx = j * window_size
            end_idx = start_idx + window_size
            
            window_data = X_norm[i, start_idx:end_idx]
            X_sliced.append(window_data)
            y_sliced.append(y[i]) # 标签复制广播
            
    X_processed = np.array(X_sliced)
    y_processed = np.array(y_sliced)
    
    return X_processed, y_processed

def process_single_edf(edf_file_path, seizure_intervals, window_sec=config.CHBMIT_WINDOW_SEC, overlap_sec=config.CHBMIT_OVERLAP_SEC, FS=config.CHBMIT_FS):
    """
    读取单个 EDF 文件，对齐通道，进行滑动切窗并打标签
    seizure_intervals: 发作时间段, e.g., [(2996, 3036)]
    window_sec: 窗口长度（秒）
    overlap_sec: 重叠长度（秒）
    """
    try:
        # preload=True 将数据载入内存以便切片
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"读取文件失败 {edf_file_path}: {e}")
        return None, None
    
    # ====================== 解决 CHB-MIT 重复通道导致 MNE 自动重命名的 Bug =====================
    rename_mapping = {}
    for ch in raw.ch_names:
        # 探测被 MNE 加上后缀的重复通道 (如 'T8-P8-0', 'T8-P8-1')
        if ch.endswith('-0') or ch.endswith('-1') or ch.endswith('-2'):
            base_name = ch[:-2] # 截断后缀，还原真实名字
            # 如果还原后的名字在白名单里，且还没有被恢复过，就把它加进重命名映射表
            if base_name in config.CHBMIT_TARGET_CHANNELS and base_name not in rename_mapping.values():
                rename_mapping[ch] = base_name
                
    if rename_mapping:
        raw.rename_channels(rename_mapping)
    # =========================================================================================

    # ================= 核心新增：临床级信号滤波去噪 =================
    # 1. 陷波滤波 (Notch Filter)
    raw.notch_filter(freqs=60.0, verbose=False)
    
    # 2. 带通滤波 (Bandpass Filter)
    raw.filter(l_freq=0.5, h_freq=50.0, verbose=False)
    # ================================================================

    # 1. 物理阉割：强制通道对齐 (解决 Channels changed 问题)
    try:
        raw.pick(config.CHBMIT_TARGET_CHANNELS)
    except ValueError as e:
        print(f"通道不匹配跳过 {edf_file_path}: {e}")
        return None, None

    # 获取底层 numpy 矩阵，维度变为 (18通道, 序列总长度)
    data = raw.get_data() 
    total_samples = data.shape[1]
    
    # 2. 滑动切窗参数计算
    window_size = int(window_sec * FS)
    step_size = int((window_sec - overlap_sec) * FS)
    
    windows = []
    labels = []
    
    # 将发作的秒数转换为采样点索引 
    seizure_indices = [(int(start * FS), int(end * FS)) for start, end in seizure_intervals]
    
    # 3. 开始滑动切分 
    for start_idx in range(0, total_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # 提取当前窗口数据 shape: (18, window_size)
        window_data = data[:, start_idx:end_idx]
        
        # 标签判定：严格规定窗口与专家标注的发作区间存在重叠即标记为发作期(1) 
        is_seizure = 0
        for s_start, s_end in seizure_indices:
            # 判断两个区间是否有交集
            if start_idx < s_end and end_idx > s_start:
                is_seizure = 1
                break
                
        windows.append(window_data)
        labels.append(is_seizure)
        
    return np.array(windows), np.array(labels)



# --- 测试代码 ---
if __name__ == "__main__":
    # 找一个带发作的文件测试，注意替换为你本地的真实路径
    test_edf = r"datasets\chbmit\chb01\chb01_03.edf" 
    
    # 这里传入刚才文本解析出来的字典中的值
    intervals = [(2996, 3036)] 
    
    print(f"开始处理 {test_edf} ...")
    # 按照开题报告，我们先测 2s 窗口，无重叠 
    X, y = process_single_edf(test_edf, intervals, window_sec=config.CHBMIT_WINDOW_SEC, overlap_sec=config.CHBMIT_OVERLAP_SEC, FS=config.CHBMIT_FS)
    
    if X is not None:
        print(f"切窗完成！")
        print(f"数据维度 X: {X.shape}") # 预期：(样本数, 18, 512)
        print(f"标签维度 y: {y.shape}")
        print(f"发现发作样本数 (ictal): {np.sum(y)}")
        print(f"发现非发作样本数 (non-ictal): {len(y) - np.sum(y)}")

# # --- 本地联调测试 ---
# if __name__ == "__main__":
#     # 模拟 data_loader 传过来的数据
#     dummy_X = np.random.randn(500, 4097)
#     dummy_y = np.random.randint(0, 2, 500)
    
#     print("开始预处理...")
#     X_new, y_new = preprocess_eeg(dummy_X, dummy_y)
    
#     print(f"预处理后数据形状: {X_new.shape}") 
#     print(f"预处理后标签形状: {y_new.shape}")