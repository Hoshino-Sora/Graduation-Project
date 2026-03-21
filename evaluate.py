import os
import torch
import numpy as np
import mne
# 导入咱们的“中枢神经”和“后勤/前线部队”
import config
from models import TCN_BiLSTM
from load_data import get_unified_dataloaders
from post_process import majority_voting_filter, extract_events, merge_close_events, filter_short_events

def evaluate_patient(patient_id="chb01", threshold=None):
    if threshold is None:
        threshold = config.PREDICT_THRESHOLD_TEST
        
    print(f"=== 开启 AI 脑电临床评估流水线 ({patient_id} 专场) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TCN_BiLSTM(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
    
    # 核心改动：加载属于这个病人的专属模型！
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', f'best_model_{patient_id}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载专属模型权重: {model_path}")
    else:
        print(f"警告：未找到 {patient_id} 的专属权重，请确认是否训练成功！")
        
    model.eval()

    # 核心改动：确保测试集里只放当前的这 1 个病人
    test_patients = [patient_id]
    dataloader = get_unified_dataloaders(
        patients_list=test_patients,
        batch_size=config.BATCH_SIZE,
        is_test=True 
    )
    
    # ==========================================
    # 3. 机器推理阶段 (生成原始 0/1 序列)
    # ==========================================
    all_predictions = []
    all_labels = []
    
    print("模型正在逐窗阅读几十个小时的脑电波，请稍候...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs.data, dim=1)
            # 爆改：使用外部传进来的动态阈值！不再用 config 里的死配置！
            predicted = (probs[:, 1] > threshold).int()
            
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
    
    # 动态绑定 2 倍 Collar 容差
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

    # 终极 API 化：把事件列表吐出去，给未来的“判卷脚本”用！
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
                # preload=False：不读数据，只读表头，速度快如闪电！
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                # raw.times[-1] 获取最后一个采样点的时间戳（也就是该文件的总秒数）
                total_seconds += raw.times[-1] 
            except Exception as e:
                # 某些损坏的 .edf 文件直接容错跳过
                print(f"文件 {filename} 头信息损坏，已跳过。原因: {e}")
                
    total_hours = total_seconds / 3600.0
    print(f"扫描完毕: 患者 {patient_id} 共有 {total_hours:.2f} 小时的有效脑电记录。")
    
    return total_hours

def calculate_clinical_metrics(real_events, ai_events, total_record_hours):
    """
    计算三大临床核心指标：灵敏度 (Sensitivity)、每小时误报率 (FD/h)、平均延迟 (Latency)
    :param real_events: 医生标注的真实发作事件列表 [{'start': 100, 'end': 150}, ...]
    :param ai_events: AI 预测的报警事件列表
    :param total_record_hours: 该病人脑电波数据的总时长（小时），用于计算 FD/h
    """
    hit_count = 0           # 成功命中的真实发作次数
    delays = []             # 每次命中的延迟时间 (秒)
    matched_ai_indices = set() # 记录哪些 AI 报警是有效的（剩下的全算作误报）

    # 1. 遍历每一个真实的医生发作记录
    for real in real_events:
        detected = False
        
        for i, ai in enumerate(ai_events):
            # 核心判断逻辑：只要 AI 的报警区间和真实发作区间有【任何重叠】，就算命中！
            if ai['start'] <= real['end'] and ai['end'] >= real['start']:
                if not detected: # 如果这个真实事件还没被认领
                    hit_count += 1
                    # 延迟时间 = AI 报警开始时间 - 真实发作开始时间
                    # （如果 AI 提前报警了，延迟可以是负数，临床上叫 anticipation）
                    delay = ai['start'] - real['start']
                    delays.append(delay)
                    detected = True
                
                # 把这个 AI 报警标记为“有功之臣”
                matched_ai_indices.add(i)

    # 2. 计算没命中的“垃圾误报”数量
    false_alarms = len(ai_events) - len(matched_ai_indices)

    # 3. 终极指标结算
    sensitivity = hit_count / len(real_events) if len(real_events) > 0 else 0.0
    fd_per_hour = false_alarms / total_record_hours if total_record_hours > 0 else 0.0
    avg_delay = np.mean(delays) if len(delays) > 0 else 0.0

    print(f"临床指标结算完成:")
    print(f"   - 事件检出率 (Sensitivity): {sensitivity*100:.2f}% ({hit_count}/{len(real_events)})")
    print(f"   - 每小时误报率 (FD/h): {fd_per_hour:.2f} 次/小时")
    print(f"   - 平均报警延迟 (Latency): {avg_delay:.2f} 秒")

    # 除了打印，我们还要把原始的“分子和分母”包装成字典扔出去！
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
    patient_to_eval = "chb01"
    print(f"=== 正在呼叫临床评估流水线 ({patient_to_eval} 专场) ===")
    
    # 1. AI 医生上阵：出具评估报告
    real_events, ai_events = evaluate_patient(patient_id=patient_to_eval)
    
    # 2. 时空雷达开启：自动测算该病人监控总时长
    total_hours = get_patient_total_hours(patient_id=patient_to_eval)
    
    # 3. 终极算分机器：量化临床指标
    if total_hours > 0:
        calculate_clinical_metrics(real_events, ai_events, total_record_hours=total_hours)
    else:
        print("无法计算 FD/h，总时长为 0！")