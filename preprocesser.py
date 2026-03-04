import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_eeg(X, y, window_size=409):
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

# --- 本地联调测试 ---
if __name__ == "__main__":
    # 模拟 data_loader 传过来的数据
    dummy_X = np.random.randn(500, 4097)
    dummy_y = np.random.randint(0, 2, 500)
    
    print("开始预处理...")
    X_new, y_new = preprocess_eeg(dummy_X, dummy_y)
    
    print(f"预处理后数据形状: {X_new.shape}") 
    print(f"预处理后标签形状: {y_new.shape}")