import os
import glob
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import config

class CHBMITDataset(Dataset):
    """
    针对 CHB-MIT 巨型 .npy 文件的“懒加载”数据集
    """
    def __init__(self, x_path, y_path):
        """
        :param x_path: 预处理后的特征文件路径 (比如 X_chb01.npy)
        :param y_path: 对应的标签文件路径 (比如 y_chb01.npy)
        """
        print(f"正在挂载数据管道: {os.path.basename(x_path)}")
        
        # mmap_mode='r' (内存映射机制)
        # 加了这句，哪怕文件有 50GB，瞬间就能“打开”，因为硬盘数据根本没有被读进内存！
        # 只有在具体索要某一个窗口时，那几 KB 的数据才会被真正抽进内存。
        self.X_disk = np.load(x_path, mmap_mode='r')
        self.y_disk = np.load(y_path, mmap_mode='r')
        
        # 校验数据长度是否对齐
        assert self.X_disk.shape[0] == self.y_disk.shape[0], "严重错误：X 和 y 的样本数对不上！"
        self.num_samples = self.X_disk.shape[0]
        print(f"挂载成功！该病历共包含 {self.num_samples} 个切窗样本。")

    def __len__(self):
        """
        PyTorch 规定动作 1：告诉 DataLoader 这个数据集总共有多长
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        PyTorch 规定动作 2：当 DataLoader 喊“给我第 idx 个数据”时，你该怎么给它
        """
        # 1. 从硬盘的内存映射中，精准抽出这一个切片 (此时这几 KB 数据才真正进入内存)
        x_window = self.X_disk[idx]
        y_label = self.y_disk[idx]
        
        # 2. 转换成 PyTorch 认识的张量 (Tensor): float32
        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        
        # PyTorch 的分类任务 (CrossEntropyLoss) 极其死板，标签必须是长整型 (long)
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        
        return x_tensor, y_tensor

def get_unified_dataloaders(patients_list, val_ratio=0.2, batch_size=64, force_positive_val=False, is_test=False):
    """
    大一统后勤发货工厂
    - is_test=False (训练模式): 执行时间轴切分，返回 (train_loader, val_loader)，默认打乱训练集。
    - is_test=True (测试模式): 拒绝切分，无损打包全量数据，返回 test_loader，绝对不打乱时间轴！
    """
    mode_name = "【终极体检】" if is_test else "【训练沙箱/全量】"
    print(f"\n后勤部接到 {mode_name} 指令！目标名单：{patients_list}")
    
    train_datasets = []
    val_or_test_datasets = [] # 训练模式下它是 Val，测试模式下它是 Test
    
    for pid in patients_list:
        data_dir = os.path.join(config.PROCESSED_DATA_PATH, pid, "win2s_ov0s")
        x_files = sorted(glob.glob(os.path.join(data_dir, "*_X.npy")))
        y_files = sorted(glob.glob(os.path.join(data_dir, "*_y.npy")))
        
        if len(x_files) == 0:
            print(f"警告：跳过 {pid}，未找到数据文件！")
            continue
            
        if is_test:
            # 测试模式：完全不切分，全盘端走！
            for x, y in zip(x_files, y_files):
                val_or_test_datasets.append(CHBMITDataset(x, y))
        else:
            # 训练模式：执行 80/20 时间轴切分
            split_idx = math.floor(len(x_files) * (1 - val_ratio))
            
            # 沙箱智能锚点保护
            if force_positive_val:
                has_seizure_in_val = False
                for y_f in y_files[split_idx:]:
                    if np.any(np.load(y_f, mmap_mode='r') == 1):
                        has_seizure_in_val = True
                        break
                while not has_seizure_in_val and split_idx > 0:
                    split_idx -= 1
                    if np.any(np.load(y_files[split_idx], mmap_mode='r') == 1):
                        print(f"    [沙箱模式] 触发智能锚点！{pid} 切分点已前移至 {split_idx}")
                        has_seizure_in_val = True

            # 装填兵力
            for x, y in zip(x_files[:split_idx], y_files[:split_idx]):
                train_datasets.append(CHBMITDataset(x, y))
            for x, y in zip(x_files[split_idx:], y_files[split_idx:]):
                val_or_test_datasets.append(CHBMITDataset(x, y))
                
    # ==========================================
    # 最终打包发货
    # ==========================================
    if is_test:
        test_dataset = ConcatDataset(val_or_test_datasets)
        # 极其致命：测试集绝对不能打乱 (shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
        print(f"测试军团发货完毕！总兵力: {len(test_dataset)} 窗 (时间轴已锁定)。\n")
        return test_loader
    else:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_or_test_datasets)
        # 训练打乱，验证不打乱
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
        print(f"训练军团发货完毕！Train: {len(train_dataset)} 窗 | Val: {len(val_dataset)} 窗。\n")
        return train_loader, val_loader

# --- 独立联调测试 (Sanity Check) ---
if __name__ == "__main__":
    print("=== 开启数据加载管道测试 ===")
    
    # ---------------------------------------------------------
    # 模拟你之前切好的 .npy 文件 (为了测试，我们现场造两个假的假装是硬盘里的文件)
    # ---------------------------------------------------------
    dummy_x_path = "dummy_X_chb01.npy"
    dummy_y_path = "dummy_y_chb01.npy"
    
    if not os.path.exists(dummy_x_path):
        print("正在生成模拟测试文件...")
        # 模拟 1000 个窗口，18通道，4097个采样点
        np.save(dummy_x_path, np.random.randn(1000, 18, 4097).astype(np.float32))
        # 模拟 1000 个标签 (0和1)
        np.save(dummy_y_path, np.random.randint(0, 2, 1000).astype(np.int8))
        print("测试文件生成完毕！")

    # 1. 实例化我们的 Dataset
    dataset = CHBMITDataset(x_path=dummy_x_path, y_path=dummy_y_path)
    
    # 2. 挂载到 DataLoader (兵工厂流水线)
    # batch_size=32 意味着每次吐出 32 个病历窗口；shuffle=True 意味着打乱顺序，防过拟合
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 3. 模拟深度学习训练过程，索要第一个 Batch
    print("\nDataLoader 正在抓取第一个 Batch...")
    for batch_X, batch_y in dataloader:
        print(f"成功抓取！")
        print(f"特征张量形状: {batch_X.shape} -> [Batch_Size, Channels, SeqLen]")
        print(f"标签张量形状: {batch_y.shape} -> [Batch_Size]")
        print(f"标签数据类型: {batch_y.dtype} (预期必须为 torch.int64 / long)")
        
        # 抓到一个 Batch 测试成功后，直接跳出循环
        break

    # 拔掉吸管：强制删除内存中的对象并回收垃圾
    del batch_X, batch_y, dataloader, dataset
    import gc; gc.collect()
        
    # 测试完顺手把假文件删了，保持环境干净
    os.remove(dummy_x_path)
    os.remove(dummy_y_path)