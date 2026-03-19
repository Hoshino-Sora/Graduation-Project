import numpy as np
import config

def majority_voting_filter(predictions, window_size=config.SMOOTHING_WINDOW):
    """
    平滑滤波 (Majority Voting)：消除模型精神分裂式的“孤立误报”和“短暂漏报”
    :param predictions: 模型输出的原始 0/1 序列，例如 [0, 1, 0, 1, 1, 1...]
    :param window_size: 滑动窗口大小 (必须是奇数，比如 5)
    :return: 平滑后的 0/1 序列
    """
    print(f"启动平滑滤波 (滑动窗口={window_size})...")
    smoothed = np.copy(predictions)
    pad_len = window_size // 2
    
    # 给首尾补 0，防止滑动窗口越界
    padded_preds = np.pad(predictions, (pad_len, pad_len), mode='constant', constant_values=0)
    
    for i in range(len(predictions)):
        # 截取当前窗口
        window = padded_preds[i : i + window_size]
        # 如果窗口里 1 的数量超过一半，中心点就判定为 1，否则为 0
        if np.sum(window) > (window_size / 2):
            smoothed[i] = 1
        else:
            smoothed[i] = 0
            
    return smoothed

def extract_events(predictions, window_duration=config.CHBMIT_WINDOW_SEC):
    """
    将离散的 0/1 序列转化为医生能看懂的“连续发作事件”报告
    :param predictions: 0/1 序列
    :param window_duration: 每个预测窗口代表的时间长度 (你的切片是 2 秒)
    :return: 包含发作起始和结束时间的事件列表
    """
    events = []
    in_seizure = False
    start_time = 0.0
    
    for i, pred in enumerate(predictions):
        # 状态突变：从 0 变 1，记录发作开始时间
        if pred == 1 and not in_seizure:
            in_seizure = True
            start_time = i * window_duration
            
        # 状态突变：从 1 变 0，记录发作结束时间，并保存该事件
        elif pred == 0 and in_seizure:
            in_seizure = False
            end_time = i * window_duration
            events.append({"start": start_time, "end": end_time, "duration": end_time - start_time})
            
    # 如果序列结束时还在发作，强制闭合事件
    if in_seizure:
        end_time = len(predictions) * window_duration
        events.append({"start": start_time, "end": end_time, "duration": end_time - start_time})
        
    return events

def merge_close_events(events, min_gap):
    """
    事件融合引擎：将距离极近的破碎报警，缝合成一次完整的临床发作。
    :param events: 提取出的原始事件列表
    :param min_gap: 允许的最大断档时间 (秒)
    """
    if not events:
        return []
    
    # 用 copy 防止修改原始数据引起混乱
    merged = [events[0].copy()]
    for current in events[1:]:
        previous = merged[-1]
        
        # 如果当前发作的起点，距离上一次发作的终点，小于等于容忍阈值
        if current['start'] - previous['end'] <= min_gap:
            # 缝合它们！更新终点和持续时间
            previous['end'] = current['end']
            previous['duration'] = previous['end'] - previous['start']
        else:
            merged.append(current.copy())
            
    return merged

def filter_short_events(events, min_duration=10.0):
    """
    终极物理超度：将持续时间极短的孤立伪影（如咬牙、眨眼）强行抹除！
    :param events: 事件列表
    :param min_duration: 最小存活时间 (秒)。小于这个时间的统统删掉。
    :return: 干净的事件列表
    """
    valid_events = []
    for ev in events:
        if ev['duration'] >= min_duration:
            valid_events.append(ev)
    return valid_events

# --- 独立联调测试 ---
if __name__ == "__main__":
    print("=== 医疗 AI 后处理引擎测试 ===")
    
    # 模拟一段极其拉胯、充满“精神分裂”的深度学习模型原始预测
    # 0 代表正常，1 代表发作。你的切片每个代表 2 秒。
    raw_predictions = np.array([
        0, 0, 0, 1, 0, 0, 0,  # <- 孤立的误报 (咬了一下牙)
        1, 1, 0, 1, 1, 1, 1,  # <- 真实的癫痫，但中间漏报了一次 (0)
        0, 0, 0, 0, 1, 0, 0   # <- 又是孤立的误报
    ])
    
    print("\n1. 原始 AI 预测结果 (零碎):")
    print(raw_predictions)
    raw_events = extract_events(raw_predictions)
    print(f"医生看到的报告: 发生 {len(raw_events)} 次发作。详细: {raw_events}")
    
    # --- 呼叫老专家上场洗数据 ---
    # window_size=3，意味着必须结合前后上下文来投票
    smoothed_predictions = majority_voting_filter(raw_predictions, window_size=3)
    
    print("\n2. 经过后处理滤波的最终结果 (平滑):")
    print(smoothed_predictions)
    smoothed_events = extract_events(smoothed_predictions)
    print(f"医生看到的报告: 发生 {len(smoothed_events)} 次发作。详细: {smoothed_events}")