import os
import glob
import numpy as np
import config
import features
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_single_file(x_path):
    """处理单个 _X.npy 文件，生成对应的 _dwt.npy 文件"""
    # 巧妙的字符串替换，把 _X.npy 变成 _dwt.npy
    dwt_path = x_path.replace('_X.npy', '_dwt.npy')
    
    # 如果已经算过了，直接跳过，支持断点续传！
    if os.path.exists(dwt_path):
        return f"跳过: {os.path.basename(dwt_path)} 已存在"
        
    try:
        # 读取时域波形数据
        X_data = np.load(x_path)
        num_samples = X_data.shape[0]
        
        # 准备一个空矩阵存放特征 (样本数, 378)
        dwt_features = np.zeros((num_samples, 378), dtype=np.float32)
        
        # 逐个窗口提取特征
        for i in range(num_samples):
            window = X_data[i]
            # 这里调用的是我们写好的老中医函数
            feat = features.extract_features_from_multichannel_window(window)
            dwt_features[i] = feat
            
        # 存入硬盘！
        np.save(dwt_path, dwt_features)
        return f"成功: {os.path.basename(dwt_path)} 生成完毕"
        
    except Exception as e:
        return f"失败: {os.path.basename(x_path)} 报错 -> {e}"

if __name__ == "__main__":
    print("=== 开启全库老中医离线特征提取工程 ===")
    
    # 找到所有的 _X.npy 文件
    search_pattern = os.path.join(config.PROCESSED_DATA_PATH, "*", "win2s_ov0s", "*_X.npy")
    all_x_files = sorted(glob.glob(search_pattern))
    
    print(f"总共发现 {len(all_x_files)} 个波形切片文件需要号脉...")
    
    # 使用多进程榨干 CPU 算力！(如果报错可以把 max_workers 改小一点比如 4 或 8)
    with ProcessPoolExecutor(max_workers=12) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(process_single_file, all_x_files), total=len(all_x_files)))
        
    print("\n=== 全部特征提取完成！===")