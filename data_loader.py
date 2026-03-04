import os
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

# --- 本地测试入口 ---
if __name__ == "__main__":
    # TODO: 把这里的路径换成你电脑上真实的Bonn数据集解压路径
    TEST_PATH = "./datasets/bonn/" 
    
    print("开始加载Bonn数据...")
    X, y = load_bonn_dataset(TEST_PATH)
    print(f"加载完成！")
    print(f"数据总形状 (Samples, TimeSteps): {X.shape}") 
    print(f"标签总形状: {y.shape}")