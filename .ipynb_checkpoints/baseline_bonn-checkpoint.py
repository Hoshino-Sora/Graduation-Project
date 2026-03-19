import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

import features 
import data_loader 
import preprocess 
import config

def run_baseline():
    print("=== Bonn 数据集传统特征工程与集成学习基线 ===")
    
    # ---------------------------------------------------------
    # 1. 加载 Bonn 原始数据
    # ---------------------------------------------------------
    print("\n1. 正在加载原始 Bonn 数据...")
    X_raw, y_raw = data_loader.load_bonn_dataset(config.BONN_DATA_PATH)
    print(f"原始数据总形状 (Samples, TimeSteps): {X_raw.shape}") 
    print(f"原始标签总形状: {y_raw.shape}")

    # 核心修正：严格防止数据泄露（Data Leakage）
    # 必须先在原始片段维度切分，然后再切窗！
    print("\n2. 划分数据集 (Train/Test Split, 阻断数据泄露)...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_raw
    )
    print(f"   -> 切分后 - 训练集原始片段数: {X_train_raw.shape[0]}, 测试集原始片段数: {X_test_raw.shape[0]}")

    # ---------------------------------------------------------
    # 3. 预处理与切窗 (Z-score + Window Slicing)
    # ---------------------------------------------------------
    print(f"\n3. 执行 Z-score 标准化与切窗 (Window Size: {config.BONN_WINDOW_SIZE}点)...")
    # 对训练集和测试集“严格隔离”地进行切窗
    X_train_sliced, y_train_sliced = preprocess.preprocess_eeg(X_train_raw, y_train_raw, window_size=config.BONN_WINDOW_SIZE)
    X_test_sliced, y_test_sliced = preprocess.preprocess_eeg(X_test_raw, y_test_raw, window_size=config.BONN_WINDOW_SIZE)
    
    print(f"   -> 切窗扩增后 - 训练集样本数: {X_train_sliced.shape[0]} 窗")
    print(f"   -> 切窗扩增后 - 测试集样本数: {X_test_sliced.shape[0]} 窗")

    # ---------------------------------------------------------
    # 4. 提取 DWT 频域特征
    # ---------------------------------------------------------
    print("\n4. 开始提取 DWT 离散小波特征...")
    print("   [处理训练集] -> ", end="")
    X_train_features = features.extract_all_features(X_train_sliced)
    print("   [处理测试集] -> ", end="")
    X_test_features = features.extract_all_features(X_test_sliced)

    # ---------------------------------------------------------
    # 5. 构建并训练集成学习模型
    # ---------------------------------------------------------
    print("\n5. 正在训练随机森林 (Random Forest) 模型...")
    # n_jobs=-1 榨干CPU所有核心加速训练
    rf_model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_train_features, y_train_sliced)

    # ---------------------------------------------------------
    # 6. 模型推理与评估出分
    # ---------------------------------------------------------
    print("\n6. 模型评估与出分...")
    y_pred = rf_model.predict(X_test_features)
    
    acc = accuracy_score(y_test_sliced, y_pred)
    # 对于不平衡医疗数据，F1 分数远比 Accuracy 重要
    f1 = f1_score(y_test_sliced, y_pred, pos_label=1, average='binary')
    
    print(f"\nBonn 基线模型最终成绩:")
    print(f"   -> Accuracy (准确率): {acc * 100:.2f}%")
    print(f"   -> F1-Score (发作期识别): {f1 * 100:.2f}%\n")
    
    print("详细分类报告 (Classification Report):")
    print(classification_report(y_test_sliced, y_pred, target_names=['正常(Non-ictal)', '发作(Ictal)']))

if __name__ == "__main__":
    run_baseline()