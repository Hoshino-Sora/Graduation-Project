import os
import gc
import torch
import config
from evaluate import evaluate_patient, get_patient_total_hours, calculate_clinical_metrics

# 核心改动：加上了 target_patients 参数
def run_global_inference(target_threshold=None, target_patients=None):
    if target_threshold is None:
        target_threshold = config.PREDICT_THRESHOLD_TEST
        
    print("\n" + "*"*20)
    
    # 核心改动：判断是全军出击还是狙击模式
    if target_patients:
        print(f"启动 [局部狙击] 推理流水线！目标: {target_patients} | 阈值: {target_threshold}")
        patients_to_run = target_patients
    else:
        print(f"启动 全库 24 人 [纯推理] 流水线！当前阈值: {target_threshold}")
        patients_to_run = [f"chb{i:02d}" for i in range(1, 25)]
    
    final_results = []
    
    # 全局池化累加器
    global_hits, global_real, global_fa, global_hours = 0, 0, 0, 0.0
    global_delay_sum, global_delay_count = 0.0, 0
    
    for test_patient in patients_to_run:
        model_path = os.path.join('outputs', 'models', f'best_model_{test_patient}.pth')
        if not os.path.exists(model_path):
            print(f"跳过 {test_patient}：未找到专属权重")
            continue
            
        print(f"正在对 {test_patient} 进行推理体检 (阈值: {target_threshold})...")
        real_events, ai_events = evaluate_patient(patient_id=test_patient, threshold=target_threshold)
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
    print(f"【阈值 {target_threshold} 终极微观临床评估报告】")
    
    micro_sens = (global_hits / global_real) * 100 if global_real > 0 else 0.0
    micro_fd = global_fa / global_hours if global_hours > 0 else 0.0
    micro_lat = global_delay_sum / global_delay_count if global_delay_count > 0 else 0.0
    
    print(f"全局微观检出率 (Micro Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})")
    print(f"全局微观误报率 (Micro FD/h): {micro_fd:.3f} 次/小时")
    print(f"全局微观延迟 (Micro Latency): {micro_lat:.2f} 秒")
    print("*"*15)
    
    # 核心改动：如果是狙击模式，生成的文件名加上 _Targeted 后缀，防止覆盖全量大表！
    suffix = "_Targeted" if target_patients else ""
    out_filename = f"Eval_Results_Thresh_{target_threshold}{suffix}.txt"
    with open(out_filename, "w") as f:
        f.write("Patient\tSensitivity(%)\tFD/h\tLatency(s)\n")
        for res in final_results:
            f.write(f"{res['patient']}\t{res['Sensitivity']*100:.2f}\t{res['FD/h']:.3f}\t{res['Latency']:.2f}\n")
            
    print(f"详细报表已成功导出至: {out_filename}\n")
    
    return micro_sens, micro_fd

if __name__ == "__main__":
    # 狙击模式用法演示：
    # 如果想跑全量，就把 target_patients 删掉或者设为 None
    # 如果想跑特定病人，就像下面这样写列表
    run_global_inference(target_threshold=0.2, target_patients=["chb15", "chb17"])