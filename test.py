from data_loader import load_bonn_dataset
from preprocesser import preprocess_eeg

if __name__ == "__main__":
    TEST_PATH = "./datasets/bonn/" 
    
    print("开始加载Bonn数据...")
    X, y = load_bonn_dataset(TEST_PATH)
    X_new, y_new = preprocess_eeg(X, y, window_size=409)
    
    print(f"预处理后数据形状: {X_new.shape}") 
    print(f"预处理后标签形状: {y_new.shape}")