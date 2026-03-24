import os
import gc
import torch
import config
from evaluate import evaluate_patient, get_patient_total_hours, calculate_clinical_metrics

# 核心新增：把 adaptive 开关和 k_factor 提拔到总控台参数里！
def run_global_inference(target_threshold=None, target_patients=None, use_adaptive=True, target_percentile=99.5):
    if target_threshold is None:
        target_threshold = config.PREDICT_THRESHOLD_TEST
        
    print("\n" + "*"*20)
    
    # 根据模式打印不同的开机语
    if use_adaptive:
        mode_str = f"[自适应分位数模式] (P={target_percentile}%)"
    else:
        mode_str = f"[传统固定阈值模式] (Thresh={target_threshold})"
        
    if target_patients:
        print(f"启动 [局部狙击] 推理流水线！目标: {target_patients} | 模式: {mode_str}")
        patients_to_run = target_patients
    else:
        print(f"启动 全库 24 人 [纯推理] 流水线！模式: {mode_str}")
        patients_to_run = [f"chb{i:02d}" for i in range(1, 25)]
    
    final_results = []
    
    global_hits, global_real, global_fa, global_hours = 0, 0, 0, 0.0
    global_delay_sum, global_delay_count = 0.0, 0
    
    for test_patient in patients_to_run:
        model_path = os.path.join('outputs', 'models', f'best_model_{test_patient}.pth')
        if not os.path.exists(model_path):
            print(f"跳过 {test_patient}：未找到专属权重")
            continue
            
        print(f"正在对 {test_patient} 进行推理体检...")
        # 核心打通：把总控台的指令传达到底层 evaluate.py！
        real_events, ai_events = evaluate_patient(
            patient_id=test_patient, 
            threshold=target_threshold, 
            use_adaptive_threshold=use_adaptive,
            target_percentile=target_percentile
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
            
            final_results.append({
                'patient': test_patient,
                'Sensitivity': sens,
                'FD/h': fd_h,
                'Latency': latency
            })
        
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # 终极结算与报表生成
    # ==========================================
    print("\n" + "*"*15)
    print(f"【{mode_str} 终极微观临床评估报告】")
    
    micro_sens = (global_hits / global_real) * 100 if global_real > 0 else 0.0
    micro_fd = global_fa / global_hours if global_hours > 0 else 0.0
    micro_lat = global_delay_sum / global_delay_count if global_delay_count > 0 else 0.0
    
    print(f"全局微观检出率 (Micro Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})")
    print(f"全局微观误报率 (Micro FD/h): {micro_fd:.3f} 次/小时")
    print(f"全局微观延迟 (Micro Latency): {micro_lat:.2f} 秒")
    print("*"*15)
    
    # 核心防混淆：动态生成科学的报表文件名！
    suffix = "_Targeted" if target_patients else ""
    if use_adaptive:
        out_filename = f"Eval_Results_Adaptive_P{target_percentile}{suffix}.txt"
    else:
        out_filename = f"Eval_Results_FixedThresh_{target_threshold}{suffix}.txt"
        
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("Patient\tSensitivity(%)\tFD/h\tLatency(s)\n")
        for res in final_results:
            f.write(f"{res['patient']}\t{res['Sensitivity']*100:.2f}\t{res['FD/h']:.3f}\t{res['Latency']:.2f}\n")
            
        f.write("\n" + "*"*15 + "\n")
        f.write(f"【{mode_str} 终极评估报告】\n")
        f.write(f"全局微观检出率 (Micro Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})\n")
        f.write(f"全局微观误报率 (Micro FD/h): {micro_fd:.3f} 次/小时\n")
        f.write(f"全局微观延迟 (Micro Latency): {micro_lat:.2f} 秒\n")
        f.write("*"*15 + "\n")
            
    print(f"详细报表及全局汇总已成功导出至: {out_filename}\n")
    
    return micro_sens, micro_fd

if __name__ == "__main__":
    # 终极优雅调用方式：
    run_global_inference(
        # target_patients=["chb15", "chb17"], 
        target_patients=None,
        use_adaptive=config.USE_ADAPTIVE, 
        target_percentile=config.TARGRT_PERCENTILE
    )