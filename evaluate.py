# evaluate.py
import os
import torch
import numpy as np
import config
from models import TCN_BiLSTM, DualBranchAttentionNet
from load_data import get_unified_dataloaders
from post_process import majority_voting_filter, extract_events, merge_close_events, filter_short_events

def evaluate_patient(patient_id="chb01", threshold=None, use_adaptive_threshold=True, target_percentile=98, model_type="dual"):
    if threshold is None:
        threshold = config.PREDICT_THRESHOLD_TEST
        
    print(f"=== 开启 AI 脑电临床评估流水线 ({patient_id} | 模式: {model_type}) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "baseline":
        model = TCN_BiLSTM(out_dim=128).to(device)
        model = torch.nn.Sequential(model, torch.nn.Linear(128, 2)).to(device)
        extract_dwt_flag = False
    else:
        model = DualBranchAttentionNet().to(device)
        extract_dwt_flag = True
    
    model_name = f'best_model_{model_type}_{patient_id}.pth' 
    model_path = os.path.join(config.MODEL_PATH, model_name)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载专属模型神装: {model_name}")
    else:
        print(f"警告：未找到 {patient_id} 的专属权重，已跳过！")
        return [], []
        
    model.eval()

    dataloader = get_unified_dataloaders(
        patients_list=[patient_id], batch_size=config.BATCH_SIZE, 
        is_test=True, extract_dwt=extract_dwt_flag
    )
    if dataloader is None: return [], []
        
    all_probs = []
    all_labels = []
    
    # 赛博探针日志本：用于记录双重注意力
    all_mod_weights = []
    all_channel_attns = [] 
    
    print("模型正在逐窗阅读脑电波，开启静默巡航...")
    with torch.no_grad():
        for batch in dataloader:
            if model_type == "baseline":
                inputs_wave, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs_wave)
            else:
                inputs_wave, inputs_dwt, labels = [b.to(device) for b in batch]
                # 关键：正确解包新的双重注意力元组！
                outputs, (channel_attn, mod_weights) = model(inputs_wave, inputs_dwt)
                
                # 收集权重到内存中
                all_mod_weights.append(mod_weights.cpu().numpy())
                all_channel_attns.append(channel_attn.cpu().numpy())
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    true_labels = np.array(all_labels)

    # ==========================================
    # 🌟 惊艳全场的可解释性报告 (XAI Report)
    # ==========================================
    if model_type == "dual" and len(all_mod_weights) > 0:
        # 沿 Batch 维度拼接并求全局平均
        avg_mod = np.concatenate(all_mod_weights, axis=0).mean(axis=0) # 形状: [2]
        avg_chan = np.concatenate(all_channel_attns, axis=0).mean(axis=0).squeeze() # 形状: [18]
        
        print("\n" + "*"*15)
        print(f"【AI 原生可解释性分析报告 (XAI)】")
        print(f"模态博弈大盘:")
        print(f"   - 左脑 (时序波形专家) 平均决策权重: {avg_mod[0]*100:.1f}%")
        print(f"   - 右脑 (频域空间神探) 平均决策权重: {avg_mod[1]*100:.1f}%")
        
        print(f"空间地形图 Top-3 核心高危通道:")
        top3_idx = np.argsort(avg_chan)[::-1][:3]
        for rank, idx in enumerate(top3_idx):
            ch_name = config.CHBMIT_TARGET_CHANNELS[idx]
            ch_score = avg_chan[idx]
            print(f"   - Rank {rank+1}: {ch_name} (全天候平均关注度: {ch_score*100:.1f}%)")
        print("*"*15 + "\n")

    # 1. TTA: 自适应及格线标定
    if use_adaptive_threshold:
        dynamic_thresh = np.percentile(all_probs, target_percentile)
        final_thresh = np.clip(dynamic_thresh, 0.01, 0.99)
        print(f"[自适应 TTA] 患者 {patient_id} 专属底噪分析:")
        print(f"   - {target_percentile}% 分位数截断阈值: {final_thresh:.4f}\n")
    else:
        final_thresh = threshold

    raw_predictions = (all_probs > final_thresh).astype(int)

    # 2. 临床后处理流水线
    smoothed_predictions = majority_voting_filter(raw_predictions, window_size=config.SMOOTHING_WINDOW)
    raw_ai_events = extract_events(smoothed_predictions, window_duration=config.CHBMIT_WINDOW_SEC)
    real_events = extract_events(true_labels, window_duration=config.CHBMIT_WINDOW_SEC)
    
    ai_events = merge_close_events(raw_ai_events, min_gap=30) 
    ai_events = filter_short_events(ai_events, min_duration=5.0)
    
    # 3. 打印精简版战报
    print(f"医生标定真实发作: {len(real_events)} 次 | AI 最终报警: {len(ai_events)} 次")
    return real_events, ai_events

def get_patient_total_hours(patient_id):
    import glob
    data_dir = os.path.join(config.PROCESSED_DATA_PATH, patient_id, "win2s_ov0s")
    y_files = glob.glob(os.path.join(data_dir, "*_y.npy"))
    if not y_files: return 0.0
        
    total_windows = sum([np.load(f, mmap_mode='r').shape[0] for f in y_files])
    return (total_windows * config.CHBMIT_WINDOW_SEC) / 3600.0

def calculate_clinical_metrics(real_events, ai_events, total_record_hours):
    hit_count = 0           
    delays = []             
    matched_ai_indices = set() 

    for real in real_events:
        detected = False
        for i, ai in enumerate(ai_events):
            # 严格的 Any-Overlap 判定
            if ai['start'] <= real['end'] and ai['end'] >= real['start']:
                if not detected: 
                    hit_count += 1
                    delays.append(ai['start'] - real['start'])
                    detected = True
                matched_ai_indices.add(i)

    false_alarms = len(ai_events) - len(matched_ai_indices)
    sensitivity = hit_count / len(real_events) if len(real_events) > 0 else 0.0
    fd_per_hour = false_alarms / total_record_hours if total_record_hours > 0 else 0.0
    avg_delay = np.mean(delays) if len(delays) > 0 else 0.0

    raw_counts = {'hit_count': hit_count, 'real_total': len(real_events),
                  'false_alarms': false_alarms, 'hours': total_record_hours,
                  'delay_sum': sum(delays), 'delay_count': len(delays)}
    return sensitivity, fd_per_hour, avg_delay, raw_counts