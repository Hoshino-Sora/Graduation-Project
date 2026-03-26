# run_eval_all.py
import os
import gc
import torch
import config
from evaluate import evaluate_patient, get_patient_total_hours, calculate_clinical_metrics

def run_global_inference(target_patients=None):
    """
    终极全库推理大流水线 (参数全部由 config.py 总控台接管)
    """
    print("\n" + "*"*40)
    
    # 从总控台读取最高指令
    use_adaptive = config.USE_ADAPTIVE
    target_percentile = config.TARGRT_PERCENTILE
    target_threshold = config.PREDICT_THRESHOLD_TEST
    model_type = "dual" if config.USE_DUAL_BRANCH else "baseline"
    
    # 动态生成作战模式代号
    if use_adaptive:
        mode_str = f"[自适应 TTA 模式] (P={target_percentile}%)"
    else:
        mode_str = f"[传统固定阈值模式] (Thresh={target_threshold})"
        
    if target_patients:
        print(f"启动 [局部狙击] 推理流水线！目标: {target_patients}")
    else:
        print(f"启动 [全库 24 人] 大阅兵！")
    print(f"挂载架构: {model_type.upper()}")
    print(f"评估模式: {mode_str}")
    print("*"*40 + "\n")
    
    patients_to_run = target_patients if target_patients else [f"chb{i:02d}" for i in range(1, 25)]
    
    final_results = []
    global_hits, global_real, global_fa, global_hours = 0, 0, 0, 0.0
    global_delay_sum, global_delay_count = 0.0, 0
    
    for test_patient in patients_to_run:
        model_path = os.path.join(config.MODEL_PATH, f'best_model_{model_type}_{test_patient}.pth')
        if not os.path.exists(model_path):
            print(f"跳过 {test_patient}：未找到专属权重 {model_path}")
            continue
            
        # 核心战术执行：调用底层 evaluate.py
        real_events, ai_events = evaluate_patient(
            patient_id=test_patient, 
            threshold=target_threshold, 
            use_adaptive_threshold=use_adaptive,
            target_percentile=target_percentile,
            model_type=model_type
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
        
        # 扫地出门，防止 OOM
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # 终极结算与科学报表生成
    # ==========================================
    if not final_results:
        print("\n警告：所有目标均未产出有效结果，流水线终止！")
        return 0.0, 0.0

    print("\n" + "*"*20)
    print(f"【{mode_str} 终极临床通关报告】")
    
    micro_sens = (global_hits / global_real) * 100 if global_real > 0 else 0.0
    micro_fd = global_fa / global_hours if global_hours > 0 else 0.0
    micro_lat = global_delay_sum / global_delay_count if global_delay_count > 0 else 0.0
    
    print(f"全局微观检出率 (Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})")
    print(f"全局微观误报率 (FD/h): {micro_fd:.3f} 次/小时")
    print(f"全局微观延迟 (Latency): {micro_lat:.2f} 秒")
    print("*"*20)
    
    # 动态生成报表文件名
    suffix = "_Targeted" if target_patients else "_ALL"
    thresh_tag = f"Adaptive_P{target_percentile}" if use_adaptive else f"Fixed_{target_threshold}"
    out_filename = f"Eval_{model_type.upper()}_{thresh_tag}{suffix}.txt"
        
    with open(out_filename, "w", encoding='utf-8') as f:
        f.write("Patient\tSensitivity(%)\tFD/h\tLatency(s)\n")
        for res in final_results:
            f.write(f"{res['patient']}\t{res['Sensitivity']*100:.2f}\t{res['FD/h']:.3f}\t{res['Latency']:.2f}\n")
            
        f.write("\n" + "*"*15 + "\n")
        f.write(f"【{mode_str} 终极评估报告】\n")
        f.write(f"全局微观检出率 (Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})\n")
        f.write(f"全局微观误报率 (FD/h): {micro_fd:.3f} 次/小时\n")
        f.write(f"全局微观延迟 (Latency): {micro_lat:.2f} 秒\n")
        f.write("*"*15 + "\n")
            
    print(f"\n详细报表及全局汇总已成功导出至: {out_filename}\n")
    
    return micro_sens, micro_fd

if __name__ == "__main__":
    # 统帅，你只需要在这里填入你想狙击的病人名单
    # 如果想跑全量 24 人，直接写 target_patients=None 即可！
    run_global_inference(
        target_patients=["chb01"]  # 可以是 ["chb06", "chb12", "chb16"] 等
    )