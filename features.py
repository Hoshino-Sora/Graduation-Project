import numpy as np
import pywt

def compute_statistics(coef):
    """
    计算单个频段（小波系数）的统计与非线性特征
    :param coef: 某一个频段的小波系数数组
    :return: 包含该频段特征的列表
    """
    # 1. 能量 (Energy): 反映该频段的剧烈程度，癫痫发作时高频能量会突增
    energy = np.sum(np.square(coef))
    
    # 2. 方差 (Variance) 与 标准差 (Standard Deviation): 反映信号的波动幅度
    variance = np.var(coef)
    std_dev = np.std(coef)
    
    # 3. 均值 (Mean)
    mean_val = np.mean(coef)
    
    # 将这4个核心标量特征打包
    return [energy, variance, std_dev, mean_val]

def extract_features_from_window(window_data, wavelet='db4', level=4):
    """
    对单段脑电窗口进行离散小波变换 (DWT) 并提取特征
    :param window_data: 一维数组，形状如 (4097,)
    :param wavelet: 小波基函数，'db4' 在脑电分析中最经典
    :param level: 分解层数。4层分解会产生 5 个频段 (1个低频近似cA4 + 4个高频细节cD4, cD3, cD2, cD1)
    :return: 提取出的一维特征向量
    """
    # 核心动作：执行 DWT 分解
    # coeffs 的结构为 [cA4, cD4, cD3, cD2, cD1]，对应脑电的不同生理频段
    coeffs = pywt.wavedec(window_data, wavelet, level=level)
    
    window_features = []
    # 遍历拆解出来的每一个频段
    for coef in coeffs:
        # 计算该频段的统计特征，并拼接到总列表中
        band_features = compute_statistics(coef)
        window_features.extend(band_features)
        
    return np.array(window_features)

def extract_all_features(X):
    """
    批量处理所有窗口数据的特征提取流水线
    :param X: 标准化后的原始脑电矩阵，形状 (Samples, TimeSteps) -> 如 Bonn 的 (5000, 4097)
    :return: 特征矩阵 (Samples, N_features)
    """
    print(f"开始提取 DWT 频域特征，总样本数: {X.shape[0]} ...")
    
    all_features = []
    for i in range(X.shape[0]):
        # 因为 Bonn 是单通道，X[i] 就是一个一维的时间序列
        feat = extract_features_from_window(X[i])
        all_features.append(feat)
        
        # 进度条
        if (i + 1) % 1000 == 0:
            print(f"   -> 已处理 {i + 1} / {X.shape[0]} 个样本...")
            
    feature_matrix = np.array(all_features)
    print(f"特征提取完成！特征矩阵维度: {feature_matrix.shape}")
    return feature_matrix

# --- 独立联调测试 ---
if __name__ == "__main__":
    # 模拟一个 Bonn 数据集中的脑电波窗口 (假设包含 4097 个采样点)
    dummy_window = np.random.randn(4097)
    
    print("测试单窗口特征提取...")
    single_feat = extract_features_from_window(dummy_window)
    # 因为 5个频段 * 每个频段4个特征 = 20 个特征
    print(f"单窗口特征维度: {single_feat.shape} (预期为 20)")
    
    print("\n测试批量特征提取...")
    dummy_X = np.random.randn(10, 4097) # 模拟 10 个样本
    X_features = extract_all_features(dummy_X)
    print(f"批量特征矩阵维度: {X_features.shape} (预期为 10x20)")