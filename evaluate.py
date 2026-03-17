import os
import torch
import numpy as np

# 导入咱们的“中枢神经”和“后勤/前线部队”
import config
from models import TCN_BiLSTM
from load_data import get_patient_dataloader
from post_process import majority_voting_filter, extract_events, merge_close_events

def evaluate_patient(patient_id="chb01"):
    print(f"=== 开启 AI 脑电临床评估流水线 ({patient_id} 专场) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 1. 组装模型并加载“最强大脑”
    # ==========================================
    model = TCN_BiLSTM(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', 'latest_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载最新模型权重: {model_path}")
    else:
        print("警告：未找到训练好的权重，模型将使用随机初始化的脑子瞎猜！")
        
    model.eval() # 极其重要：关闭 Dropout，开启测试模式

    # ==========================================
    # 2. 挂载测试数据管道 (集团军发货)
    # ==========================================
    # 生死红线：shuffle=False！绝对不能打乱时间轴！
    dataloader = get_patient_dataloader(patient_id=patient_id, 
                                        batch_size=config.BATCH_SIZE, 
                                        shuffle=False)
    
    # ==========================================
    # 3. 机器推理阶段 (生成原始 0/1 序列)
    # ==========================================
    all_predictions = []
    all_labels = []
    
    print("模型正在逐窗阅读几十个小时的脑电波，请稍候...")
    with torch.no_grad(): # 不计算梯度，省显存提速
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Argmax：赢家通吃
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    raw_predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)

    # ==========================================
    # 4. 临床后处理阶段 (老专家 + 缝合大师介入)
    # ==========================================
    print("后处理平滑引擎介入，清洗孤立误报...")
    smoothed_predictions = majority_voting_filter(raw_predictions, window_size=config.SMOOTHING_WINDOW)
    
    print("正在提取初始发作事件...")
    raw_ai_events = extract_events(smoothed_predictions, window_duration=config.CHBMIT_WINDOW_SEC)
    real_events = extract_events(true_labels, window_duration=config.CHBMIT_WINDOW_SEC)
    
    # 🌟 你的神级发明：动态绑定 1.5 倍 Collar 容差
    fusion_gap = 1.5 * config.COLLAR_TOLERANCE
    print(f"启动宏观事件融合引擎 (容忍断档 <= {fusion_gap}秒)...")
    ai_events = merge_close_events(raw_ai_events, min_gap=fusion_gap)
    
    # ==========================================
    # 5. 打印最终临床体检报告
    # ==========================================
    print("\n" + "="*40)
    print(f"[{patient_id} 全时段临床评估报告]")
    print("="*40)
    print(f"医生标定的真实发作次数: {len(real_events)} 次")
    for idx, ev in enumerate(real_events):
        print(f"   - 真实事件 {idx+1}: 第 {ev['start']} 秒 -> 第 {ev['end']} 秒 (持续 {ev['duration']}s)")
        
    print(f"\nAI 最终报警次数 (融合后): {len(ai_events)} 次")
    for idx, ev in enumerate(ai_events):
        print(f"   - 报警事件 {idx+1}: 第 {ev['start']} 秒 -> 第 {ev['end']} 秒 (持续 {ev['duration']}s)")
    print("="*40)

    # 终极 API 化：把事件列表吐出去，给未来的“判卷脚本”用！
    return real_events, ai_events

if __name__ == "__main__":
    # 一键呼叫 chb01 的全套体检！
    evaluate_patient(patient_id="chb01")