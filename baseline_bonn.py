import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import features # 导入 DWT 特征提取模块
import data_loader # 导入数据读取模块
import config

def run_baseline():
    print("=== 阶段三：Bonn 数据集传统特征工程与集成学习基线 ===")
    
    # ---------------------------------------------------------
    # 1. 加载 Bonn 数据
    # ---------------------------------------------------------
    print("\n1. 正在加载并预处理 Bonn 数据...")
    TEST_PATH = config.BONN_DATA_PATH 
        
    print("开始加载Bonn数据...")
    X_raw, y_raw = data_loader.load_bonn_dataset(TEST_PATH)
    print(f"加载完成！")
    print(f"数据总形状 (Samples, TimeSteps): {X_raw.shape}") 
    print(f"标签总形状: {y_raw.shape}")



    # ---------------------------------------------------------
    # 2. 提取 DWT 频域特征
    # ---------------------------------------------------------
    print("\n2. 开始提取 DWT 离散小波特征...")
    # 调用我们的神仙辅助模块，把 4097 维的波形压缩成 20 维的黄金特征
    X_features = features.extract_all_features(X_raw)

    # ---------------------------------------------------------
    # 3. 划分训练集与测试集
    # ---------------------------------------------------------
    print("\n3. 划分数据集 (Train/Test Split)...")
    # 按照 80% 训练，20% 测试的黄金比例划分
    # stratify=y_raw 极其重要：确保切分后，训练集和测试集里的正负样本比例保持一致
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"   -> 训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    # ---------------------------------------------------------
    # 4. 构建并训练集成学习模型
    # ---------------------------------------------------------
    print("\n4. 正在训练随机森林 (Random Forest) 模型...")
    # n_estimators=100: 种 100 棵决策树来进行投票
    # n_jobs=-1: 火力全开，调用你电脑 CPU 的所有核心来加速训练
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. 模型推理与评估出分
    # ---------------------------------------------------------
    print("\n5. 模型评估与出分...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBonn 基线模型 Accuracy (准确率): {acc * 100:.2f}%\n")
    
    print("详细分类报告 (Classification Report):")
    # target_names 翻译成医学标签
    print(classification_report(y_test, y_pred, target_names=['正常(Non-ictal)', '发作(Ictal)']))

if __name__ == "__main__":
    run_baseline()