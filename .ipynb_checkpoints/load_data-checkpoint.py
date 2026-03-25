import os
import glob
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import config
import sys

# 🌟 核心新增：导入我们的老中医特征提取器！
import features

class CHBMITDataset(Dataset):
    """
    针对 CHB-MIT 巨型 .npy 文件的“懒加载”数据集 (升级版：自带老中医 DWT 特征提取)
    """
    def __init__(self, x_path, y_path, extract_dwt=True):
        self.extract_dwt = extract_dwt
        print(f"正在挂载数据管道: {os.path.basename(x_path)}")
        
        self.X_disk = np.load(x_path, mmap_mode='r')
        self.y_disk = np.load(y_path, mmap_mode='r')
        self.num_samples = self.X_disk.shape[0]
        
        if self.extract_dwt:
            dwt_path = x_path.replace('_X.npy', '_dwt.npy')
            self.dwt_disk = np.load(dwt_path)
            assert self.dwt_disk.shape[0] == self.num_samples, "特征数与波形数不对齐！"
            
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 从硬盘抽取原始切片
        x_window = self.X_disk[idx].copy()
        y_label = self.y_disk[idx].copy()
        
        # 2. 转换为 PyTorch 张量
        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        
        # 核心升级：如果开启了老中医模式，当场计算 DWT 物理特征！
        if self.extract_dwt:
            dwt_window = self.dwt_disk[idx].copy()
            dwt_tensor = torch.tensor(dwt_window, dtype=torch.float32)
            return x_tensor, dwt_tensor, y_tensor
            
        return x_tensor, y_tensor

def get_unified_dataloaders(patients_list, val_ratio=0.2, batch_size=64, force_positive_val=False, is_test=False, extract_dwt=True):
    """
    大一统后勤发货工厂 (新增 extract_dwt 开关，默认开启)
    """
    mode_name = "【终极体检】" if is_test else "【训练沙箱/全量】"
    print(f"\n后勤部接到 {mode_name} 指令！目标名单：{patients_list}")
    
    train_datasets = []
    val_or_test_datasets = [] 
    
    for pid in patients_list:
        data_dir = os.path.join(config.PROCESSED_DATA_PATH, pid, "win2s_ov0s")
        x_files = sorted(glob.glob(os.path.join(data_dir, "*_X.npy")))
        y_files = sorted(glob.glob(os.path.join(data_dir, "*_y.npy")))
        
        if len(x_files) == 0:
            continue
            
        if is_test:
            for x, y in zip(x_files, y_files):
                val_or_test_datasets.append(CHBMITDataset(x, y, extract_dwt=extract_dwt))
        else:
            split_idx = math.floor(len(x_files) * (1 - val_ratio))
            
            if force_positive_val:
                has_seizure_in_val = False
                for y_f in y_files[split_idx:]:
                    if np.any(np.load(y_f, mmap_mode='r') == 1):
                        has_seizure_in_val = True
                        break
                while not has_seizure_in_val and split_idx > 0:
                    split_idx -= 1
                    if np.any(np.load(y_files[split_idx], mmap_mode='r') == 1):
                        has_seizure_in_val = True

            for x, y in zip(x_files[:split_idx], y_files[:split_idx]):
                train_datasets.append(CHBMITDataset(x, y, extract_dwt=extract_dwt))
            for x, y in zip(x_files[split_idx:], y_files[split_idx:]):
                val_or_test_datasets.append(CHBMITDataset(x, y, extract_dwt=extract_dwt))
                
    # ==========================================
    # 最终打包发货 (带空载安全锁)
    # ==========================================
    num_workers = 12 if sys.platform.startswith('linux') else 0

    if is_test:
        # 核心拦截锁 1：如果什么数据都没找到，直接返回 None！
        if len(val_or_test_datasets) == 0:
            print(f"后勤部警告：{patients_list} 仓库为空，无法发货！")
            return None
            
        test_dataset = ConcatDataset(val_or_test_datasets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return test_loader
    else:
        # 核心拦截锁 2：同理保护训练模式
        if len(train_datasets) == 0 or len(val_or_test_datasets) == 0:
            print(f"后勤部警告：训练集或验证集兵力为空，无法发货！")
            return None, None
            
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_or_test_datasets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

# --- 独立联调测试 (Sanity Check) ---
if __name__ == "__main__":
    print("=== 开启双分支数据加载管道测试 ===")
    
    dummy_x_path = "dummy_X_chb01.npy"
    dummy_y_path = "dummy_y_chb01.npy"
    
    if not os.path.exists(dummy_x_path):
        print("正在生成模拟测试文件...")
        # 模拟 1000 个窗口，18通道，512个采样点 (CHB-MIT 常见尺寸)
        np.save(dummy_x_path, np.random.randn(1000, 18, 512).astype(np.float32))
        np.save(dummy_y_path, np.random.randint(0, 2, 1000).astype(np.int8))
        print("测试文件生成完毕！")

    # 1. 实例化 Dataset，默认开启老中医特征提取
    dataset = CHBMITDataset(x_path=dummy_x_path, y_path=dummy_y_path, extract_dwt=config.EXTRACT_DWT)
    
    # 2. 挂载到 DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 3. 抓取第一个 Batch 测试
    print("\nDataLoader 正在抓取第一个 Batch (会在此刻实时计算 DWT)...")
    
    # 注意接收值的变化：现在解包出来是 3 个变量！
    for batch_X, batch_dwt, batch_y in dataloader:
        print(f"成功抓取三件套！")
        print(f"黑盒时域波形张量: {batch_X.shape} -> [Batch, Channels, SeqLen]")
        print(f"白盒频域特征张量: {batch_dwt.shape} -> [Batch, DWT_Features]")
        print(f"标签张量形状: {batch_y.shape} -> [Batch]")
        break

    del batch_X, batch_dwt, batch_y, dataloader, dataset
    import gc; gc.collect()
        
    os.remove(dummy_x_path)
    os.remove(dummy_y_path)