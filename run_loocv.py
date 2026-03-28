# run_loocv.py
import os
import gc
import time
import torch
import config
from train import train_model
from evaluate import evaluate_patient, get_patient_total_hours, calculate_clinical_metrics

def run_loocv_pipeline():
    print(f"\n启动 V2.0 LOOCV 全自动化流水线 (模式: {'Dual-Branch' if config.USE_DUAL_BRANCH else 'Baseline'})\n")
    pipeline_start_time = time.time()
    
    FULL_PATIENTS_LIST = [f"chb{i:02d}" for i in range(1, 25)]
    target_patients = config.TARGET_PATIENTS # 可以随时放开全员
    
    final_results = []
    global_hits, global_real, global_fa, global_hours, global_delay_sum, global_delay_count = 0, 0, 0, 0.0, 0.0, 0
    
    for test_patient in target_patients:
        train_patients = [p for p in FULL_PATIENTS_LIST if p != test_patient]
        
        # 1. 点火训练
        train_model(test_patient, train_patients)
        
        # 2. 临床评估
        current_model_type = "dual" if config.USE_DUAL_BRANCH else "baseline"
        real_events, ai_events = evaluate_patient(
            patient_id=test_patient,
            use_adaptive_threshold=config.USE_ADAPTIVE,
            target_percentile=config.TARGRT_PERCENTILE,
            model_type=current_model_type
        )
        total_hours = get_patient_total_hours(patient_id=test_patient)
        
        if total_hours > 0:
            sens, fd_h, latency, raw = calculate_clinical_metrics(real_events, ai_events, total_hours)
            global_hits += raw['hit_count']
            global_real += raw['real_total']
            global_fa += raw['false_alarms']
            global_hours += raw['hours']
            global_delay_sum += raw['delay_sum']
            global_delay_count += raw['delay_count']

            final_results.append({'patient': test_patient, 'Sensitivity': sens, 'FD/h': fd_h, 'Latency': latency})
            
        torch.cuda.empty_cache()
        gc.collect()
        
    # 3. 终极报表
    print("\n" + "*"*30)
    print("【LOOCV 终极微观临床评估大通关！】")
    micro_sens = (global_hits / global_real) * 100 if global_real > 0 else 0.0
    micro_fd = global_fa / global_hours if global_hours > 0 else 0.0
    micro_lat = global_delay_sum / global_delay_count if global_delay_count > 0 else 0.0
    
    print(f"全局检出率 (Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})")
    print(f"全局误报率 (FD/h): {micro_fd:.3f} 次/小时")
    print(f"全局延迟 (Latency): {micro_lat:.2f} 秒")
    print("*"*30 + "\n")

if __name__ == "__main__":
    run_loocv_pipeline()