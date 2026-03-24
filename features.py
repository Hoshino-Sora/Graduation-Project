import numpy as np
import pywt

def compute_wavelet_entropy(energies):
    """
    新增神级特征：计算小波能量熵 (Wavelet Energy Entropy)
    反映大脑的混乱程度。癫痫发作时神经元高度同步，熵值会显著下降。
    """
    total_energy = np.sum(energies)
    if total_energy == 0:
        return 0.0
    
    # 计算每个频段的能量占比 (概率分布)
    probs = energies / total_energy
    
    # 过滤掉概率为0的项，防止 log2(0) 报错
    probs = probs[probs > 0]
    
    # 香农熵公式：- sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def extract_features_from_multichannel_window(window_data, wavelet='db4', level=4):
    """
    对【多通道】脑电窗口进行 DWT 并提取物理特征
    :param window_data: 形状为 (Num_Channels, TimeSteps) -> 例如 CHB-MIT 的 (18, 512)
    :param wavelet: 小波基函数 'db4'
    :param level: 分解层数
    :return: 拍平的一维特征向量
    """
    num_channels = window_data.shape[0]
    all_channel_features = []
    
    # 遍历每一个电极通道
    for ch in range(num_channels):
        channel_signal = window_data[ch]
        
        # DWT 分解：产生 5 个频段 (cA4, cD4, cD3, cD2, cD1)
        coeffs = pywt.wavedec(channel_signal, wavelet, level=level)
        
        ch_features = []
        energies = []
        
        # 提取各个频段的统计特征
        for coef in coeffs:
            energy = np.sum(np.square(coef))
            variance = np.var(coef)
            std_dev = np.std(coef)
            mean_val = np.mean(coef)
            
            ch_features.extend([energy, variance, std_dev, mean_val])
            energies.append(energy)
            
        # 算完该通道所有频段能量后，追加计算小波熵
        entropy = compute_wavelet_entropy(np.array(energies))
        ch_features.append(entropy)
        
        # 将该通道的所有特征拼接到总列表中
        all_channel_features.extend(ch_features)
        
    # 返回一个极其硬核的特征向量
    return np.array(all_channel_features, dtype=np.float32)

# --- 独立联调测试 ---
if __name__ == "__main__":
    # 模拟 CHB-MIT 的一个 2 秒切窗：18 个通道，256Hz采样率 -> 512 个采样点
    dummy_channels = 18
    dummy_timesteps = 512
    dummy_window = np.random.randn(dummy_channels, dummy_timesteps)
    
    print("测试多通道 DWT 特征提取...")
    features = extract_features_from_multichannel_window(dummy_window)
    
    # 算笔账：5个频段 * 每个频段4个特征 = 20 个特征
    # 追加 1 个小波熵 = 21 个特征/通道
    # 18 个通道 * 21 = 378 个特征
    print(f"输入窗口维度: {dummy_window.shape}")
    print(f"输出先验物理特征维度: {features.shape} (预期为 18 * 21 = 378)")