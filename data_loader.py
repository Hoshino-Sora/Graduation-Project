import os
import re
import numpy as np
import glob

def load_bonn_dataset(base_path):
    """
    解析 Bonn 数据集
    :param base_path: 数据集根目录，应包含 A, B, C, D, E 五个子文件夹
    :return: data (numpy array), labels (numpy array)
    """
    # 定义类别映射：这里我们将 A,B,C,D 设为0 (非发作)，E设为1 (发作)
    # 这是典型的二分类任务设定。如果你要做三分类或五分类，在这里改逻辑
    folder_mapping = {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1} 
    # 如果你的文件夹叫 Z, O, N, F, S，请自行修改上面的字典键值
    
    all_data = []
    all_labels = []
    
    for folder, label in folder_mapping.items():
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"警告: 找不到文件夹 {folder_path}，请检查路径。")
            continue
            
        # 获取文件夹下所有 txt 文件
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        for file in txt_files:
            try:
                # 读取一维数据，4097个点
                signal = np.loadtxt(file)
                
                # 内存泄漏/数据脏乱排查：确保每个文件绝对是4097个点
                if signal.shape[0] != 4097:
                    print(f"数据异常警告: 文件 {file} 的长度为 {signal.shape[0]}，跳过。")
                    continue
                    
                all_data.append(signal)
                all_labels.append(label)
                
            except Exception as e:
                print(f"读取文件 {file} 时报错: {e}")
                
    # 将列表转换为 numpy array，便于后续送入 PyTorch
    data_array = np.array(all_data)
    label_array = np.array(all_labels)
    
    return data_array, label_array

def parse_summary_file(summary_path):
    """
    解析 CHB-MIT 的 summary.txt 文件，提取发作时间戳。
    返回格式: { 'chb01_01.edf': [], 'chb01_03.edf': [(2996, 3036)] }
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"找不到 Summary 文件: {summary_path}")
        
    with open(summary_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # 利用 "File Name:" 作为切分定界符，将文本块打散
    blocks = re.split(r'File Name:\s*', text)[1:]
    
    meta_index = {}
    
    for block in blocks:
        lines = block.strip().split('\n')
        filename = lines[0].strip()
        
        # 提取发作次数
        num_seizures_match = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if not num_seizures_match:
            continue
            
        num_seizures = int(num_seizures_match.group(1))
        seizures = []
        
        if num_seizures > 0:
            # 核心防坑：兼容 "Seizure Start Time" 和 "Seizure 1 Start Time" 两种写法
            starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)', block)
            ends = re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)', block)
            
            # 维度冲突警告：如果提取到的起止时间数量不匹配，说明正则表达式漏掉了数据，必须抛出异常
            if len(starts) != len(ends) or len(starts) != num_seizures:
                raise ValueError(f"解析 {filename} 时时间戳对齐失败！检测到 {num_seizures} 次发作，但只提取到 {len(starts)} 个起点。")
                
            for s, e in zip(starts, ends):
                seizures.append((int(s), int(e)))
                
        meta_index[filename] = seizures
        
    return meta_index

# --- 测试代码 ---
if __name__ == "__main__":
    # 假设你的 summary 文件就在当前目录测试
    # 实际项目中应引入 config.py 的路径
    test_path = "datasets\chbmit\chb01\chb01-summary.txt" 
    try:
        index_dict = parse_summary_file(test_path)
        print("解析成功！共提取 EDF 文件数:", len(index_dict))
        print("带发作的文件示例 (chb01_03):", index_dict.get('chb01_03.edf', '未找到'))
    except Exception as e:
        print(f"报错了：{e}")

# # --- 本地测试入口 ---
# if __name__ == "__main__":
#     # TODO: 把这里的路径换成你电脑上真实的Bonn数据集解压路径
#     TEST_PATH = "./datasets/bonn/" 
    
#     print("开始加载Bonn数据...")
#     X, y = load_bonn_dataset(TEST_PATH)
#     print(f"加载完成！")
#     print(f"数据总形状 (Samples, TimeSteps): {X.shape}") 
#     print(f"标签总形状: {y.shape}")