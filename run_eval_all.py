import os
import gc
import torch
import config
from evaluate import evaluate_patient, get_patient_total_hours, calculate_clinical_metrics

def run_global_inference(target_threshold=None):
    if target_threshold is None:
        target_threshold = config.PREDICT_THRESHOLD_TEST
        
    print("\n" + "*"*20)
    print(f"启动全库 24 人 [纯推理] 流水线！当前阈值: {target_threshold}")
    
    all_patients = [f"chb{i:02d}" for i in range(1, 25)]
    
    # 新增：用来收集 24 个人详细报表的列表
    final_results = []
    
    # 全局池化累加器
    global_hits, global_real, global_fa, global_hours = 0, 0, 0, 0.0
    global_delay_sum, global_delay_count = 0.0, 0
    
    for test_patient in all_patients:
        model_path = os.path.join('outputs', 'models', f'best_model_{test_patient}.pth')
        if not os.path.exists(model_path):
            print(f"跳过 {test_patient}：未找到专属权重")
            continue
            
        print(f"正在对 {test_patient} 进行推理体检 (阈值: {target_threshold})...")
        real_events, ai_events = evaluate_patient(patient_id=test_patient, threshold=target_threshold)
        total_hours = get_patient_total_hours(patient_id=test_patient)
        
        if total_hours > 0:
            sens, fd_h, latency, raw = calculate_clinical_metrics(real_events, ai_events, total_hours)
            
            # 全局累加
            global_hits += raw['hit_count']
            global_real += raw['real_total']
            global_fa += raw['false_alarms']
            global_hours += raw['hours']
            global_delay_sum += raw['delay_sum']
            global_delay_count += raw['delay_count']
            
            # 新增：把单人成绩记录进字典
            final_results.append({
                'patient': test_patient,
                'Sensitivity': sens,
                'FD/h': fd_h,
                'Latency': latency
            })
        
        # 物理洗锅
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # 终极结算与报表生成
    # ==========================================
    print("\n" + "*"*15)
    print(f"【阈值 {target_threshold} 终极微观临床评估报告】")
    
    micro_sens = (global_hits / global_real) * 100 if global_real > 0 else 0.0
    micro_fd = global_fa / global_hours if global_hours > 0 else 0.0
    micro_lat = global_delay_sum / global_delay_count if global_delay_count > 0 else 0.0
    
    print(f"全局微观检出率 (Micro Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})")
    print(f"全局微观误报率 (Micro FD/h): {micro_fd:.3f} 次/小时")
    print(f"全局微观延迟 (Micro Latency): {micro_lat:.2f} 秒")
    print("*"*15)
    
    # 新增：将大表动态保存为带有阈值标记的 txt 文件
    out_filename = f"Eval_Results_Thresh_{target_threshold}.txt"
    with open(out_filename, "w") as f:
        f.write("Patient\tSensitivity(%)\tFD/h\tLatency(s)\n")
        for res in final_results:
            f.write(f"{res['patient']}\t{res['Sensitivity']*100:.2f}\t{res['FD/h']:.3f}\t{res['Latency']:.2f}\n")
            
    print(f"详细单人报表已成功导出至: {out_filename}\n")
    
    return micro_sens, micro_fd

if __name__ == "__main__":
    # 你可以一次性测好几个你想看的阈值，它会给你生成一排 txt 文件！
    test_thresholds = [0.2] 
    for t in test_thresholds:
        run_global_inference(target_threshold=t)