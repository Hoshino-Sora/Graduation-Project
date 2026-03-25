import os
import torch
import numpy as np
import mne
import config
# 核心改动 1：把新老模型都请进指挥部！
from models import TCN_BiLSTM, DualBranchAttentionNet
from load_data import get_unified_dataloaders
from post_process import majority_voting_filter, extract_events, merge_close_events, filter_short_events

# 核心改动 2：增加 model_type 参数，默认是 'dual'，但你可以传 'baseline'
def evaluate_patient(patient_id="chb01", threshold=None, use_adaptive_threshold=True, target_percentile=98, model_type="dual"):
    if threshold is None:
        threshold = config.PREDICT_THRESHOLD_TEST
        
    print(f"=== 开启 AI 脑电临床评估流水线 ({patient_id} | 模式: {model_type}) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 核心改动 3：根据指令，挂载不同的武器和装甲！
    if model_type == "baseline":
        model = TCN_BiLSTM(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
        extract_dwt_flag = False # 老基线不需要物理特征
    else:
        model = DualBranchAttentionNet(num_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES, dwt_feature_dim=378).to(device)
        extract_dwt_flag = True  # 新神装必须开启老中医特征
    
    # 彻底抛弃兼容老名字的幻想，严格区分新老权重！
    model_name = f'best_model_{model_type}_{patient_id}.pth' 
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'models', model_name)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载专属模型权重: {model_path}")
    else:
        print(f"警告：未找到 {patient_id} 的专属权重！")
        return [], []
        
    model.eval()

    test_patients = [patient_id]
    dataloader = get_unified_dataloaders(
        patients_list=test_patients,
        batch_size=config.BATCH_SIZE,
        is_test=True,
        extract_dwt=extract_dwt_flag # 动态决定要不要提取特征
    )
    
    if dataloader is None:
        print(f"数据缺失，已安全跳过 {patient_id} 的体检。")
        return [], []
        
    all_probs = []
    all_labels = []
    
    print("模型正在逐窗阅读脑电波，请稍候...")
    with torch.no_grad():
        # 核心改动 4：根据不同模型，动态分发粮草
        for batch_idx, batch in dataloader:
            if model_type == "baseline":
                # 基线模式：后勤只吐出两件套
                inputs_wave, labels = batch
                inputs_wave = inputs_wave.to(device)
                outputs = model(inputs_wave) # 注意：你原来基线如果加了 return_features，这里要确保不传
            else:
                # 双分支模式：后勤吐出三件套
                inputs_wave, inputs_dwt, labels = batch
                inputs_wave = inputs_wave.to(device)
                inputs_dwt = inputs_dwt.to(device)
                outputs, attn_weights = model(inputs_wave, inputs_dwt)
                # 临时打印一下这一个 Batch 的注意力平均分配情况
                if batch_idx % 100 == 0:
                    print(f"   [抽查 Batch {batch_idx}] 左脑信任度: {attn_weights[:, 0].mean().item():.3f} | 右脑信任度: {attn_weights[:, 1].mean().item():.3f}")
            
            probs = torch.softmax(outputs.data, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    true_labels = np.array(all_labels)

    # ==========================================
    # 核心修正：基于非参数化分位数的自适应阈值
    # ==========================================
    if use_adaptive_threshold:
        dynamic_thresh = np.percentile(all_probs, target_percentile)
        final_thresh = np.clip(dynamic_thresh, 0.01, 0.99)
        
        print(f"\n[自适应阈值标定] 患者 {patient_id} 专属底噪分析:")
        print(f"   - {target_percentile}% 分位数计算结果: {dynamic_thresh:.4f}")
        print(f"   - 截断后最终使用阈值: {final_thresh:.4f}\n")
    else:
        final_thresh = threshold
        print(f"\n[固定阈值模式] 使用预设死板阈值: {final_thresh:.4f}\n")

    raw_predictions = (all_probs > final_thresh).astype(int)

    # ==========================================
    # 4. 临床后处理阶段
    # ==========================================
    print("后处理平滑引擎介入，清洗孤立误报...")
    smoothed_predictions = majority_voting_filter(raw_predictions, window_size=config.SMOOTHING_WINDOW)
    
    print("正在提取初始发作事件...")
    raw_ai_events = extract_events(smoothed_predictions, window_duration=config.CHBMIT_WINDOW_SEC)
    real_events = extract_events(true_labels, window_duration=config.CHBMIT_WINDOW_SEC)
    
    fusion_gap = 60
    print(f"启动宏观事件融合引擎 (容忍断档 <= {fusion_gap}秒)...")
    ai_events = merge_close_events(raw_ai_events, min_gap=fusion_gap)

    print("启动物理超度：清除持续时间不足 5 秒的孤立肌电伪影...")
    ai_events = filter_short_events(ai_events, min_duration=5.0)
    
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
    全 npy 化极速版：不再依赖原始 .edf 文件，直接通过统计 .npy 切片的数量来反推总时长！
    极其优雅，零内存消耗，瞬间出结果！
    """
    # 直接去咱们的后勤加工厂找数据
    data_dir = os.path.join(config.PROCESSED_DATA_PATH, patient_id, "win2s_ov0s")
    import glob
    y_files = glob.glob(os.path.join(data_dir, "*_y.npy"))
    
    if not y_files:
        print(f"找不到患者 {patient_id} 的预处理标签文件，无法计算时长！")
        return 0.0
        
    print(f"正在极速扫描 {patient_id} 的 .npy 时空碎片...")
    
    total_windows = 0
    for y_file in y_files:
        # 使用 mmap_mode='r' 极速读取，根本不进内存，只看形状！
        y_data = np.load(y_file, mmap_mode='r')
        total_windows += y_data.shape[0]
        
    # 计算物理总时间：窗口数 * 每个窗口的秒数 (默认2秒)
    total_seconds = total_windows * config.CHBMIT_WINDOW_SEC
    total_hours = total_seconds / 3600.0
    
    print(f"扫描完毕: 患者 {patient_id} 共有 {total_windows} 个切窗，折合 {total_hours:.2f} 小时的有效脑电记录。")
    
    return total_hours

def calculate_clinical_metrics(real_events, ai_events, total_record_hours):
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
    
    # 核心联动：单点调试也保持绝对同步
    current_model_type = "dual" if config.USE_DUAL_BRANCH else "baseline"
    
    print(f"=== 正在呼叫临床评估流水线 ({patient_to_eval} 专场 | 模式: {current_model_type}) ===")
    
    real_events, ai_events = evaluate_patient(
        patient_id=patient_to_eval, 
        use_adaptive_threshold=True,
        model_type=current_model_type  # 传参！
    )
    total_hours = get_patient_total_hours(patient_id=patient_to_eval)
    
    if total_hours > 0:
        calculate_clinical_metrics(real_events, ai_events, total_record_hours=total_hours)
    else:
        print("无法计算 FD/h，总时长为 0！")