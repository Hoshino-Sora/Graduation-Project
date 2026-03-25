import os
import gc
import time  # 新增：时间记录模块
import torch
import numpy as np

# 导入你的核心模块
from train import train_model
from evaluate import evaluate_patient, get_patient_total_hours, calculate_clinical_metrics

def run_loocv_pipeline():
    print("启动 [留一法交叉验证 (LOOCV)] 全自动化基线评估大流水线！")
    
    # 按下总秒表！
    pipeline_start_time = time.time()
    
    FULL_PATIENTS_LIST = [f"chb{i:02d}" for i in range(1, 25)]
    # target_patients = [f"chb{i:02d}" for i in range(1, 25)]
    # target_patients = ["chb06", "chb12", "chb13", "chb14", "chb16"]
    target_patients = ["chb16"]
    
    # 用来收集所有人的成绩表
    final_results = []

    # 建立全局累加器 (这就是你要的分子和分母！)
    global_hits = 0
    global_real = 0
    global_fa = 0
    global_hours = 0.0
    global_delay_sum = 0.0
    global_delay_count = 0
    
    for test_patient in target_patients:
        print("\n" + "*"*25)
        print(f"正在攻坚靶标: {test_patient}")
        print("*"*25)
        
        # 1. 划定训练集团军 (除目标外的 23 个人)
        train_patients = [p for p in FULL_PATIENTS_LIST if p != test_patient]
        
        # 2. 启动该病人的专属炼丹炉
        train_model(test_patient, train_patients)
        
        # 3. 训练完毕，立刻启动该病人的临床体检！
        print(f"\n训练完成！正在对 {test_patient} 进行全时段临床体检...")
        real_events, ai_events = evaluate_patient(patient_id=test_patient)
        total_hours = get_patient_total_hours(patient_id=test_patient)
        
        if total_hours > 0:
            sens, fd_h, latency, raw = calculate_clinical_metrics(real_events, ai_events, total_hours)
            
            # 疯狂往全局池子里倒数据 (分子加分子，分母加分母)
            global_hits += raw['hit_count']
            global_real += raw['real_total']
            global_fa += raw['false_alarms']
            global_hours += raw['hours']
            global_delay_sum += raw['delay_sum']
            global_delay_count += raw['delay_count']

            # 记录成绩
            final_results.append({
                'patient': test_patient,
                'Sensitivity': sens,
                'FD/h': fd_h,
                'Latency': latency
            })
        else:
            print(f"{test_patient} 脑电时长异常，已跳过指标计算。")
            
        # 4. 极其重要：工业级显存大清洗
        torch.cuda.empty_cache()
        gc.collect()
        
    # ==========================================
    # 5. 终极结算大表输出 (论文核心表格数据！)
    # ==========================================
    print("\n" + "*"*15)
    print("【全库 24 人 LOOCV 终极微观临床评估大通关！】")
    
    # 分子比分母！防极端值偏倚的完美算法！
    micro_sens = (global_hits / global_real) * 100 if global_real > 0 else 0.0
    micro_fd = global_fa / global_hours if global_hours > 0 else 0.0
    micro_lat = global_delay_sum / global_delay_count if global_delay_count > 0 else 0.0
    
    print(f"全局微观检出率 (Micro Sensitivity): {micro_sens:.2f}% ({global_hits}/{global_real})")
    print(f"全局微观误报率 (Micro FD/h): {micro_fd:.3f} 次/小时 (共 {global_fa} 次误报 / {global_hours:.1f} 小时)")
    print(f"全局微观延迟 (Micro Latency): {micro_lat:.2f} 秒")
    print("*"*15)
    
    # 把详细的 list 保存到 txt 里，方便直接贴到 Excel 里画图！
    with open("LOOCV_Baseline_Results.txt", "w") as f:
        f.write("Patient\tSensitivity(%)\tFD/h\tLatency(s)\n")
        for res in final_results:
            f.write(f"{res['patient']}\t{res['Sensitivity']*100:.2f}\t{res['FD/h']:.3f}\t{res['Latency']:.2f}\n")

    # 结算总耗时并华丽输出
    pipeline_end_time = time.time()
    total_seconds = pipeline_end_time - pipeline_start_time
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "*"*10)
    print(f"本次 LOOCV 24 人全量流水线总运行时长: {int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒")
    print("*"*10 + "\n")

if __name__ == "__main__":
    run_loocv_pipeline()