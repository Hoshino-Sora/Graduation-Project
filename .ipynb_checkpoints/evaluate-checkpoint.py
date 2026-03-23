import os
import torch
import numpy as np
import mne
# 导入咱们的“中枢神经”和“后勤/前线部队”
import config
from models import TCN_BiLSTM
from load_data import get_unified_dataloaders
from post_process import majority_voting_filter, extract_events, merge_close_events, filter_short_events

# 核心新增：加入了 use_adaptive_threshold 和 k_factor 参数
def evaluate_patient(patient_id="chb01", threshold=None, use_adaptive_threshold=True, target_percentile=99.5):
    if threshold is None:
        threshold = config.PREDICT_THRESHOLD_TEST
        
    print(f"=== 开启 AI 脑电临床评估流水线 ({patient_id} 专场) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TCN_BiLSTM(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
    
    # 加载专属模型
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', f'best_model_{patient_id}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载专属模型权重: {model_path}")
    else:
        print(f"警告：未找到 {patient_id} 的专属权重，请确认是否训练成功！")
        
    model.eval()

    test_patients = [patient_id]
    dataloader = get_unified_dataloaders(
        patients_list=test_patients,
        batch_size=config.BATCH_SIZE,
        is_test=True 
    )
    
    # ==========================================
    # 3. 机器推理阶段 (先号脉，存概率，不急着下定论！)
    # ==========================================
    all_probs = []
    all_labels = []
    
    print("模型正在逐窗阅读几十个小时的脑电波，请稍候...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs.data, dim=1)
            # 爆改：直接把发作概率存进列表，暂时不做 0/1 截断！
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    true_labels = np.array(all_labels)

    # ==========================================
    # 核心修正：基于非参数化分位数的自适应阈值
    # ==========================================
    if use_adaptive_threshold:
        # 直接使用外部传入的 target_percentile 控制严格程度
        dynamic_thresh = np.percentile(all_probs, target_percentile)
        
        # 物理安全锁：即便底噪再高，阈值也不能低于 0.2，最高不超过 0.9
        final_thresh = np.clip(dynamic_thresh, 0.20, 0.90)
        
        print(f"\n[自适应阈值标定] 患者 {patient_id} 专属底噪分析:")
        print(f"   - {target_percentile}% 分位数计算结果: {dynamic_thresh:.4f}")
        print(f"   - 截断后最终使用阈值: {final_thresh:.4f}\n")
    else:
        final_thresh = threshold
        print(f"\n[固定阈值模式] 使用预设死板阈值: {final_thresh:.4f}\n")

    # 根据量身定制的 final_thresh 统一宣判
    raw_predictions = (all_probs > final_thresh).astype(int)

    # ==========================================
    # 4. 临床后处理阶段 (老专家 + 缝合大师介入)
    # ==========================================
    print("后处理平滑引擎介入，清洗孤立误报...")
    smoothed_predictions = majority_voting_filter(raw_predictions, window_size=config.SMOOTHING_WINDOW)
    
    print("正在提取初始发作事件...")
    raw_ai_events = extract_events(smoothed_predictions, window_duration=config.CHBMIT_WINDOW_SEC)
    real_events = extract_events(true_labels, window_duration=config.CHBMIT_WINDOW_SEC)
    
    fusion_gap = 2 * config.COLLAR_TOLERANCE
    print(f"启动宏观事件融合引擎 (容忍断档 <= {fusion_gap}秒)...")
    ai_events = merge_close_events(raw_ai_events, min_gap=fusion_gap)

    print("启动物理超度：清除持续时间不足 10 秒的孤立肌电伪影...")
    ai_events = filter_short_events(ai_events, min_duration=10.0)
    
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

    return real_events, ai_events


def get_patient_total_hours(patient_id):
    """
    动态扫描计算单个患者所有有效 EDF 文件的总时长 (小时)。
    利用 preload=False 黑科技，只读文件头，零内存消耗，极速秒出！
    """
    edf_folder = os.path.join(config.CHBMIT_DATA_PATH, patient_id)
    total_seconds = 0.0
    
    if not os.path.exists(edf_folder):
        print(f"找不到患者目录: {edf_folder}")
        return 0.0
        
    print(f"正在极速扫描 {patient_id} 的脑电时空记录...")
    
    for filename in os.listdir(edf_folder):
        if filename.endswith('.edf'):
            edf_path = os.path.join(edf_folder, filename)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                total_seconds += raw.times[-1] 
            except Exception as e:
                print(f"文件 {filename} 头信息损坏，已跳过。原因: {e}")
                
    total_hours = total_seconds / 3600.0
    print(f"扫描完毕: 患者 {patient_id} 共有 {total_hours:.2f} 小时的有效脑电记录。")
    
    return total_hours

def calculate_clinical_metrics(real_events, ai_events, total_record_hours):
    """
    计算三大临床核心指标：灵敏度 (Sensitivity)、每小时误报率 (FD/h)、平均延迟 (Latency)
    """
    hit_count = 0           
    delays = []             
    matched_ai_indices = set() 

    for real in real_events:
        detected = False
        
        for i, ai in enumerate(ai_events):
            if ai['start'] <= real['end'] and ai['end'] >= real['start']:
                if not detected: 
                    hit_count += 1
                    delay = ai['start'] - real['start']
                    delays.append(delay)
                    detected = True
                matched_ai_indices.add(i)

    false_alarms = len(ai_events) - len(matched_ai_indices)

    sensitivity = hit_count / len(real_events) if len(real_events) > 0 else 0.0
    fd_per_hour = false_alarms / total_record_hours if total_record_hours > 0 else 0.0
    avg_delay = np.mean(delays) if len(delays) > 0 else 0.0

    print(f"临床指标结算完成:")
    print(f"   - 事件检出率 (Sensitivity): {sensitivity*100:.2f}% ({hit_count}/{len(real_events)})")
    print(f"   - 每小时误报率 (FD/h): {fd_per_hour:.2f} 次/小时")
    print(f"   - 平均报警延迟 (Latency): {avg_delay:.2f} 秒")

    raw_counts = {
        'hit_count': hit_count,
        'real_total': len(real_events),
        'false_alarms': false_alarms,
        'hours': total_record_hours,
        'delay_sum': sum(delays),
        'delay_count': len(delays)
    }
    
    return sensitivity, fd_per_hour, avg_delay, raw_counts

if __name__ == "__main__":
    patient_to_eval = "chb15"
    print(f"=== 正在呼叫临床评估流水线 ({patient_to_eval} 专场) ===")
    
    real_events, ai_events = evaluate_patient(patient_id=patient_to_eval, use_adaptive_threshold=True)
    total_hours = get_patient_total_hours(patient_id=patient_to_eval)
    
    if total_hours > 0:
        calculate_clinical_metrics(real_events, ai_events, total_record_hours=total_hours)
    else:
        print("无法计算 FD/h，总时长为 0！")