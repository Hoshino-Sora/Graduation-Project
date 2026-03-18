import os
import numpy as np
import config
import time
from data_loader import parse_summary_file
from preprocess import process_single_edf

def build_chbmit_dataset(summary_file_path, edf_folder_path, output_dir):
    """
    全量处理引擎：解析 summary，遍历 edf，切窗并持久化保存到硬盘
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"开始解析索引文件: {summary_file_path}")
    meta_index = parse_summary_file(summary_file_path)
    
    total_files = len(meta_index)
    processed_count = 0
    
    for filename, intervals in meta_index.items():
        edf_path = os.path.join(edf_folder_path, filename)
        
        # 跳过不存在的文件
        if not os.path.exists(edf_path):
            print(f"文件未找到，跳过: {edf_path}")
            continue
            
        print(f"[{processed_count+1}/{total_files}] 正在处理 {filename} ...")
        
        # 调用核心切窗引擎
        X, y = process_single_edf(edf_path, intervals)
        
        if X is not None and y is not None:
            # 命名规范：以原文件名为前缀，保存 X 和 y
            base_name = filename.replace('.edf', '')
            x_save_path = os.path.join(output_dir, f"{base_name}_X.npy")
            y_save_path = os.path.join(output_dir, f"{base_name}_y.npy")
            
            # 使用 numpy 持久化落盘
            np.save(x_save_path, X)
            np.save(y_save_path, y)
            
            print(f"    -> 已落盘: {x_save_path} (形状: {X.shape})")
        
        processed_count += 1
        
    print(f"\n该患者的全量数据预处理完成！共处理 {processed_count} 个文件。")

# --- 批处理启动入口 ---
if __name__ == "__main__":
    # 动态参数命名输出路径
    win_sec = config.CHBMIT_WINDOW_SEC
    ov_sec = config.CHBMIT_OVERLAP_SEC
    folder_name = f"win{win_sec}s_ov{ov_sec}s"
    
    # 核心控制台：想要跑全量，修改这个列表即可
    # target_patients = ['chb01', 'chb02', 'chb04'] 
    
    # 全量 24 个病人的列表生成式 (留作备用，需要跑全库时取消注释)：
    target_patients = [f"chb{i:02d}" for i in range(1, 4)] 
    
    print(f"启动 CHB-MIT 全库批量切片流水线 (参数: 窗口={win_sec}s, 重叠={ov_sec}s)...")
    print(f"待处理患者列表: {target_patients}\n" + "="*50)
    
    start_time = time.time()
    
    for patient_id in target_patients:
        print(f"\n>>> 正在启动流水线: {patient_id} <<<")
        
        # 动态拼接每个病人的输入输出路径
        SUMMARY_PATH = os.path.join(config.CHBMIT_DATA_PATH, patient_id, f"{patient_id}-summary.txt")
        EDF_FOLDER = os.path.join(config.CHBMIT_DATA_PATH, patient_id)
        OUTPUT_FOLDER = os.path.join(config.PROCESSED_DATA_PATH, patient_id, folder_name)
        
        # 如果这个病人的原始文件夹不存在，直接跳过
        if not os.path.exists(EDF_FOLDER):
            print(f"找不到患者目录 {EDF_FOLDER}，跳过该患者。")
            continue
            
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # 调用函数
        build_chbmit_dataset(SUMMARY_PATH, EDF_FOLDER, OUTPUT_FOLDER)
        
    total_time = (time.time() - start_time) / 60
    print(f"\n全库处理指令下达完毕！总耗时: {total_time:.2f} 分钟。")