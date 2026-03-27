import os
import glob
import numpy as np
import config

def extract_seizures_from_npy(patients_list, window_sec=config.CHBMIT_WINDOW_SEC):
    """
    【极简核心】从 _y.npy 中提取目标患者群体的真实发作区间
    """
    seizure_dict = {}
    for pid in patients_list:
        data_dir = os.path.join(config.PROCESSED_DATA_PATH, pid, "win2s_ov0s")
        y_files = sorted(glob.glob(os.path.join(data_dir, "*_y.npy")))
        if not y_files: 
            continue
            
        # 懒加载并拼接该病人的所有标签
        y_all = [np.load(y_f, mmap_mode='r') for y_f in y_files]
        if not y_all:
            continue
        y_all = np.concatenate(y_all)
        
        # 边缘检测：寻找 0->1(上升沿) 和 1->0(下降沿)
        padded_y = np.pad(y_all, (1, 1), 'constant')
        diffs = np.diff(padded_y)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        # 转换为秒数并保存
        seizures = [(s * window_sec, e * window_sec) for s, e in zip(starts, ends)]
        if seizures:
            seizure_dict[pid] = seizures
            
    return seizure_dict

def calculate_set_stats(seizure_dict):
    """
    【极简核心】仅计算并返回发作时长的 1% 和 99% 分位数软边界
    """
    all_durations = []
    
    for pid, seizures in seizure_dict.items():
        for start_sec, end_sec in seizures:
            all_durations.append(end_sec - start_sec)
            
    if not all_durations:
        return None
        
    return {
        'p01_dur': np.percentile(all_durations, 1),
        'p99_dur': np.percentile(all_durations, 99)
    }