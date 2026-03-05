'''
    输入：
        多个CSV文件，每个文件包含节点的时间序列数据，代表：真值、本模型的预测值、其他基准模型的预测值

    思路：
        从多个CSV文件中读取指定node_id的数据
        按照时间顺序排列，对每个时间节点所处的交通状态进行判断
        将各个模型的预测状态与真值状态进行对比
        计算精确率(Precision)、召回率(Recall)、F1-score
        
        新增功能：
        - 状态转换事件识别：检测"变拥堵"和"变畅通"事件
        - 转换事件匹配：使用时间窗口和最近邻原则匹配真值和预测事件
        - 转换事件指标：计算精准率、召回率、平均延迟时间

    输出：
        两个CSV文件：
        1. 状态识别的精确率、召回率、F1-score统计
        2. 转换事件识别的精准率、召回率、平均延迟时间统计
'''

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from typing import List, Dict, Tuple

# 定义交通状态
class TrafficState:
    FREE_FLOW = 0          # 自由流
    QUEUE_FORMING = 1      # 排队形成
    QUEUE_DISSIPATING = 2  # 排队消散
    SATURATED = 3          # 饱和流
    QUEUED = 4             # 排队状态
    UNKNOWN = 5            # 未知（无数据）

# 状态名称映射
STATE_NAMES = {
    TrafficState.FREE_FLOW: '自由流',
    TrafficState.QUEUE_FORMING: '排队形成',
    TrafficState.QUEUE_DISSIPATING: '排队消散',
    TrafficState.SATURATED: '饱和流',
    TrafficState.QUEUED: '排队状态',
    TrafficState.UNKNOWN: '无数据'
}

# 转换事件类型
class TransitionEventType:
    TO_CONGESTION = 'to_congestion'  # 变拥堵
    TO_FREE_FLOW = 'to_free_flow'    # 变畅通

# 简化状态分类（用于转换事件检测）
class SimplifiedState:
    FREE_FLOW = 'free_flow'      # 自由流（包括状态0、5）
    CONGESTION = 'congestion'    # 拥堵流（包括状态1、2、3）
    QUEUED = 'queued'            # 排队状态（状态4）

def map_to_simplified_state(state: int) -> str:
    """将原始状态映射到简化状态"""
    if state in [TrafficState.FREE_FLOW, TrafficState.UNKNOWN]:
        return SimplifiedState.FREE_FLOW
    elif state in [TrafficState.QUEUE_FORMING, TrafficState.QUEUE_DISSIPATING, TrafficState.SATURATED]:
        return SimplifiedState.CONGESTION
    elif state == TrafficState.QUEUED:
        return SimplifiedState.QUEUED
    else:
        return SimplifiedState.FREE_FLOW  # 默认归为自由流

def interpolate_missing_values(speeds):
    """对缺失值进行线性插值"""
    speeds_series = pd.Series(speeds)
    speeds_interpolated = speeds_series.interpolate(method='linear', limit_direction='both')
    return speeds_interpolated.values

def calculate_pairwise_slopes(speeds):
    """计算两两之间的斜率"""
    slopes = []
    for i in range(len(speeds) - 1):
        slope = speeds[i+1] - speeds[i]
        slopes.append(slope)
    return slopes

def classify_state_by_voting(speed, future_speeds, slope_threshold=1.5, 
                             lower_threshold=5, upper_threshold=25):
    """通过投票方式分类交通状态"""
    if pd.isna(speed):
        return TrafficState.FREE_FLOW
    
    if speed > upper_threshold:
        return TrafficState.FREE_FLOW
    
    if speed <= lower_threshold:
        return TrafficState.QUEUED
    
    if len(future_speeds) < 2:
        return TrafficState.SATURATED
    
    interpolated_speeds = interpolate_missing_values(future_speeds)
    slopes = calculate_pairwise_slopes(interpolated_speeds)
    
    if len(slopes) == 0:
        return TrafficState.SATURATED
    
    vote_counts = {
        TrafficState.FREE_FLOW: 0,
        TrafficState.QUEUE_FORMING: 0,
        TrafficState.QUEUE_DISSIPATING: 0,
        TrafficState.SATURATED: 0,
        TrafficState.QUEUED: 0
    }
    
    for slope in slopes:
        if slope < -slope_threshold:
            vote_counts[TrafficState.QUEUE_FORMING] += 1
        elif slope > slope_threshold:
            vote_counts[TrafficState.QUEUE_DISSIPATING] += 1
        else:
            vote_counts[TrafficState.SATURATED] += 1
    
    max_votes = max(vote_counts[TrafficState.QUEUE_FORMING],
                   vote_counts[TrafficState.QUEUE_DISSIPATING],
                   vote_counts[TrafficState.SATURATED])
    
    if vote_counts[TrafficState.QUEUE_FORMING] == max_votes:
        return TrafficState.QUEUE_FORMING
    elif vote_counts[TrafficState.QUEUE_DISSIPATING] == max_votes:
        return TrafficState.QUEUE_DISSIPATING
    else:
        return TrafficState.SATURATED

def estimate_states(df, node_id, value_column='avg_speed', window=10, slope_threshold=1.5,
                   lower_threshold=5, upper_threshold=25):
    """估计时间序列的交通状态"""
    node_data = df[df['node_id'] == node_id].copy()
    node_data = node_data.sort_values('time')
    
    if len(node_data) == 0:
        return pd.DataFrame(columns=['time', 'state'])
    
    if value_column not in node_data.columns:
        print(f"警告：列 '{value_column}' 不存在")
        return pd.DataFrame(columns=['time', 'state'])
    
    node_data['speed'] = node_data[value_column]
    
    states = []
    times = node_data['time'].values
    speeds = node_data['speed'].values
    
    for i in range(len(times)):
        current_speed = speeds[i]
        future_speeds = speeds[i:min(i+window+1, len(speeds))]
        
        state = classify_state_by_voting(current_speed, future_speeds, 
                                        slope_threshold=slope_threshold,
                                        lower_threshold=lower_threshold, 
                                        upper_threshold=upper_threshold)
        states.append(state)
    
    result = pd.DataFrame({
        'time': times,
        'state': states
    })
    
    return result

class TransitionEvent:
    """转换事件类"""
    def __init__(self, timestamp: float, event_type: str, from_state: str, to_state: str):
        self.timestamp = timestamp
        self.event_type = event_type  # 'to_congestion' 或 'to_free_flow'
        self.from_state = from_state
        self.to_state = to_state
        self.matched = False  # 是否已被匹配
    
    def __repr__(self):
        return f"TransitionEvent(t={self.timestamp}, type={self.event_type}, {self.from_state}->{self.to_state})"

def detect_transition_events(states_df: pd.DataFrame) -> List[TransitionEvent]:
    """检测状态转换事件
    
    参数:
        states_df: DataFrame，包含 'time' 和 'state' 列
    
    返回:
        转换事件列表
    """
    events = []
    
    if len(states_df) < 2:
        return events
    
    # 确保按时间排序
    states_df = states_df.sort_values('time').reset_index(drop=True)
    
    # 映射到简化状态
    simplified_states = [map_to_simplified_state(s) for s in states_df['state'].values]
    times = states_df['time'].values
    
    # 检测转换
    for i in range(len(simplified_states) - 1):
        current_state = simplified_states[i]
        next_state = simplified_states[i + 1]
        timestamp = times[i + 1]  # 使用转换后的时间点
        
        # 检测"变拥堵"事件
        # 1. 自由流 -> 拥堵流
        if current_state == SimplifiedState.FREE_FLOW and next_state == SimplifiedState.CONGESTION:
            events.append(TransitionEvent(timestamp, TransitionEventType.TO_CONGESTION, 
                                         current_state, next_state))
        
        # 2. 自由流 -> 排队
        elif current_state == SimplifiedState.FREE_FLOW and next_state == SimplifiedState.QUEUED:
            events.append(TransitionEvent(timestamp, TransitionEventType.TO_CONGESTION, 
                                         current_state, next_state))
        
        # 3. 拥堵流 -> 排队（这是拥堵加剧，也算变拥堵事件）
        elif current_state == SimplifiedState.CONGESTION and next_state == SimplifiedState.QUEUED:
            events.append(TransitionEvent(timestamp, TransitionEventType.TO_CONGESTION, 
                                         current_state, next_state))
        
        # 检测"变畅通"事件
        # 排队 -> 任何其他状态
        elif current_state == SimplifiedState.QUEUED and next_state != SimplifiedState.QUEUED:
            events.append(TransitionEvent(timestamp, TransitionEventType.TO_FREE_FLOW, 
                                         current_state, next_state))
    
    return events

def match_transition_events(gt_events: List[TransitionEvent], 
                           pred_events: List[TransitionEvent], 
                           time_window: float) -> Tuple[List[Tuple], List[TransitionEvent], List[TransitionEvent]]:
    """匹配真值事件和预测事件
    
    参数:
        gt_events: 真值事件列表
        pred_events: 预测事件列表
        time_window: 匹配时间窗口（单位：时间步）
    
    返回:
        (匹配对列表, 漏报事件列表, 误报事件列表)
        匹配对格式: (gt_event, pred_event, time_delay)
    """
    # 重置所有事件的匹配状态
    for event in gt_events:
        event.matched = False
    for event in pred_events:
        event.matched = False
    
    matched_pairs = []
    
    # 按时间排序
    gt_events_sorted = sorted(gt_events, key=lambda e: e.timestamp)
    pred_events_sorted = sorted(pred_events, key=lambda e: e.timestamp)
    
    # 遍历每个真值事件
    for gt_event in gt_events_sorted:
        if gt_event.matched:
            continue
        
        # 查找时间窗口内的候选预测事件
        candidates = []
        for pred_event in pred_events_sorted:
            time_diff = abs(pred_event.timestamp - gt_event.timestamp)
            
            # 在时间窗口内且事件类型匹配
            if time_diff <= time_window and pred_event.event_type == gt_event.event_type:
                candidates.append((pred_event, time_diff))
        
        if not candidates:
            continue  # 没有候选，这个gt_event将成为漏报
        
        # 按时间差排序，选择最近的
        candidates.sort(key=lambda x: x[1])
        
        # 尝试匹配最近的未匹配预测事件
        for pred_event, time_diff in candidates:
            if not pred_event.matched:
                # 成功匹配
                gt_event.matched = True
                pred_event.matched = True
                time_delay = pred_event.timestamp - gt_event.timestamp  # 正值表示预测延迟，负值表示提前
                matched_pairs.append((gt_event, pred_event, time_delay))
                break
            else:
                # 这个pred_event已经被匹配了，检查是否需要重新分配
                # 找到之前匹配这个pred_event的gt_event
                previous_match = None
                for pair in matched_pairs:
                    if pair[1] == pred_event:
                        previous_match = pair
                        break
                
                if previous_match:
                    previous_gt, _, previous_delay = previous_match
                    # 比较哪个gt_event离pred_event更近
                    if time_diff < abs(previous_delay):
                        # 当前gt_event更近，重新分配
                        previous_gt.matched = False
                        gt_event.matched = True
                        matched_pairs.remove(previous_match)
                        time_delay = pred_event.timestamp - gt_event.timestamp
                        matched_pairs.append((gt_event, pred_event, time_delay))
                        break
    
    # 统计漏报和误报
    false_negatives = [e for e in gt_events if not e.matched]
    false_positives = [e for e in pred_events if not e.matched]
    
    return matched_pairs, false_negatives, false_positives

def calculate_transition_metrics(gt_events: List[TransitionEvent], 
                                pred_events: List[TransitionEvent], 
                                time_window: float) -> Dict:
    """计算转换事件的精准率、召回率、平均延迟
    
    参数:
        gt_events: 真值事件列表
        pred_events: 预测事件列表
        time_window: 匹配时间窗口
    
    返回:
        包含各类指标的字典
    """
    # 分别处理两种事件类型
    results = {}
    
    for event_type in [TransitionEventType.TO_CONGESTION, TransitionEventType.TO_FREE_FLOW]:
        # 筛选特定类型的事件
        gt_type_events = [e for e in gt_events if e.event_type == event_type]
        pred_type_events = [e for e in pred_events if e.event_type == event_type]
        
        # 匹配事件
        matched_pairs, false_negatives, false_positives = match_transition_events(
            gt_type_events, pred_type_events, time_window
        )
        
        # 计算指标
        tp = len(matched_pairs)  # 真正例
        fn = len(false_negatives)  # 假负例（漏报）
        fp = len(false_positives)  # 假正例（误报）
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算平均延迟（只考虑匹配成功的事件）
        if matched_pairs:
            delays = [delay for _, _, delay in matched_pairs]
            avg_delay = np.mean(delays)
            std_delay = np.std(delays)
            median_delay = np.median(delays)
            # 计算绝对平均延迟（不考虑方向）
            abs_delays = [abs(delay) for delay in delays]
            avg_abs_delay = np.mean(abs_delays)
            median_abs_delay = np.median(abs_delays)
        else:
            avg_delay = 0.0
            std_delay = 0.0
            median_delay = 0.0
            avg_abs_delay = 0.0
            median_abs_delay = 0.0
        
        event_name = '变拥堵' if event_type == TransitionEventType.TO_CONGESTION else '变畅通'
        
        results[event_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_negatives': fn,
            'false_positives': fp,
            'avg_delay': avg_delay,
            'std_delay': std_delay,
            'median_delay': median_delay,
            'avg_abs_delay': avg_abs_delay,
            'median_abs_delay': median_abs_delay,
            'gt_count': len(gt_type_events),
            'pred_count': len(pred_type_events)
        }
    
    # 计算总体指标
    total_tp = sum(results[k]['true_positives'] for k in results)
    total_fn = sum(results[k]['false_negatives'] for k in results)
    total_fp = sum(results[k]['false_positives'] for k in results)
    
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
    
    # 计算总体平均延迟
    all_delays = []
    for event_type in [TransitionEventType.TO_CONGESTION, TransitionEventType.TO_FREE_FLOW]:
        gt_type_events = [e for e in gt_events if e.event_type == event_type]
        pred_type_events = [e for e in pred_events if e.event_type == event_type]
        matched_pairs, _, _ = match_transition_events(gt_type_events, pred_type_events, time_window)
        all_delays.extend([delay for _, _, delay in matched_pairs])
    
    if all_delays:
        total_avg_delay = np.mean(all_delays)
        total_std_delay = np.std(all_delays)
        total_median_delay = np.median(all_delays)
        # 计算总体绝对平均延迟
        all_abs_delays = [abs(delay) for delay in all_delays]
        total_avg_abs_delay = np.mean(all_abs_delays)
        total_median_abs_delay = np.median(all_abs_delays)
    else:
        total_avg_delay = 0.0
        total_std_delay = 0.0
        total_median_delay = 0.0
        total_avg_abs_delay = 0.0
        total_median_abs_delay = 0.0
    
    results['总体'] = {
        'precision': total_precision,
        'recall': total_recall,
        'f1': total_f1,
        'true_positives': total_tp,
        'false_negatives': total_fn,
        'false_positives': total_fp,
        'avg_delay': total_avg_delay,
        'std_delay': total_std_delay,
        'median_delay': total_median_delay,
        'avg_abs_delay': total_avg_abs_delay,
        'median_abs_delay': total_median_abs_delay,
        'gt_count': len(gt_events),
        'pred_count': len(pred_events)
    }
    
    return results

def calculate_metrics(ground_truth_states, predicted_states):
    """计算精确率、召回率、F1-score
    
    参数:
        ground_truth_states: 真值状态列表
        predicted_states: 预测状态列表
    
    返回:
        字典，包含各类别和总体的精确率、召回率、F1-score
    """
    # 确保两个列表长度相同
    min_len = min(len(ground_truth_states), len(predicted_states))
    ground_truth_states = ground_truth_states[:min_len]
    predicted_states = predicted_states[:min_len]
    
    # 计算各类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth_states, 
        predicted_states, 
        labels=[TrafficState.FREE_FLOW, TrafficState.QUEUE_FORMING, 
                TrafficState.QUEUE_DISSIPATING, TrafficState.SATURATED, 
                TrafficState.QUEUED],
        average=None,
        zero_division=0
    )
    
    # 计算加权平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        ground_truth_states, 
        predicted_states,
        labels=[TrafficState.FREE_FLOW, TrafficState.QUEUE_FORMING, 
                TrafficState.QUEUE_DISSIPATING, TrafficState.SATURATED, 
                TrafficState.QUEUED],
        average='weighted',
        zero_division=0
    )
    
    # 计算宏平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        ground_truth_states, 
        predicted_states,
        labels=[TrafficState.FREE_FLOW, TrafficState.QUEUE_FORMING, 
                TrafficState.QUEUE_DISSIPATING, TrafficState.SATURATED, 
                TrafficState.QUEUED],
        average='macro',
        zero_division=0
    )
    
    # 组织结果
    results = {
        'states': {
            STATE_NAMES[TrafficState.FREE_FLOW]: {
                'precision': precision[0],
                'recall': recall[0],
                'f1': f1[0],
                'support': support[0]
            },
            STATE_NAMES[TrafficState.QUEUE_FORMING]: {
                'precision': precision[1],
                'recall': recall[1],
                'f1': f1[1],
                'support': support[1]
            },
            STATE_NAMES[TrafficState.QUEUE_DISSIPATING]: {
                'precision': precision[2],
                'recall': recall[2],
                'f1': f1[2],
                'support': support[2]
            },
            STATE_NAMES[TrafficState.SATURATED]: {
                'precision': precision[3],
                'recall': recall[3],
                'f1': f1[3],
                'support': support[3]
            },
            STATE_NAMES[TrafficState.QUEUED]: {
                'precision': precision[4],
                'recall': recall[4],
                'f1': f1[4],
                'support': support[4]
            }
        },
        'weighted_avg': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        },
        'macro_avg': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        }
    }
    
    return results

def main():
    # ========== 配置参数 ==========
    NODE_ID = 4
    start_frame = 250
    end_frame = 600
    output_path = f'../data/draw/d210191000/melt/node{NODE_ID}_fscore_metrics.csv'
    transition_output_path = f'../data/draw/d210191000/melt/node{NODE_ID}_transition_metrics.csv'
    
    # 转换事件匹配窗口（时间步）
    TRANSITION_TIME_WINDOW = 30  # 可调整参数
    
    # ========== 数据配置 ==========
    # 真值配置
    ground_truth_config = {
        'file_path': '../data/draw/d210191000/d210291000_lane_node_stats.csv',
        'label': '真值',
        'value_column': 'avg_speed',
        'window': 5,
        'slope_threshold': 0.155,
        'lower_threshold': 5,
        'upper_threshold': 17
    }
    
    # 预测模型配置
    prediction_configs = [
        {
            'file_path': '../data/draw/d210191000/melt/0302l3/inference_results_L3.csv',
            'label': '本研究模型',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 17
        },
        {
            'file_path': '../data/draw/d210191000/melt/0303stgnnl2l3/hybrid_simple_stgnn_l2l3_predictions.csv',
            'label': 'STGNN',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 17
        },
        {
            'file_path': '../data/draw/d210191000/melt/0302l2/inference_results_L2.csv',
            'label': 'L2',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 17
        },
        {
            'file_path': '../data/draw/d210191000/melt/0302l1/inference_results_L1.csv',
            'label': 'L1',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 17
        }
    ]
    
    # ========== 加载真值数据 ==========
    print("加载真值数据...")
    try:
        df_gt = pd.read_csv(ground_truth_config['file_path'])
        print(f"真值数据: {len(df_gt)} 条记录")
        
        ground_truth_states_df = estimate_states(
            df_gt, NODE_ID, 
            value_column=ground_truth_config['value_column'],
            window=ground_truth_config['window'],
            slope_threshold=ground_truth_config['slope_threshold'],
            lower_threshold=ground_truth_config['lower_threshold'],
            upper_threshold=ground_truth_config['upper_threshold']
        )
        
        # 限制时间范围
        ground_truth_states_df = ground_truth_states_df[
            (ground_truth_states_df['time'] >= start_frame) & 
            (ground_truth_states_df['time'] <= end_frame)
        ].copy()
        
        if len(ground_truth_states_df) == 0:
            print(f"错误：未找到node_id={NODE_ID}的真值数据！")
            return
        
        print(f"真值状态: {len(ground_truth_states_df)} 个时间点")
        
    except FileNotFoundError:
        print(f"错误：真值文件 {ground_truth_config['file_path']} 不存在！")
        return
    except Exception as e:
        print(f"错误：处理真值文件时出错: {e}")
        return
    
    # ========== 计算各模型的指标 ==========
    print("\n计算各模型的F-score指标...")
    results_list = []
    transition_results_list = []
    
    # 检测真值的转换事件
    print("\n检测真值转换事件...")
    gt_events = detect_transition_events(ground_truth_states_df)
    print(f"真值转换事件: {len(gt_events)} 个")
    gt_to_congestion = [e for e in gt_events if e.event_type == TransitionEventType.TO_CONGESTION]
    gt_to_free_flow = [e for e in gt_events if e.event_type == TransitionEventType.TO_FREE_FLOW]
    print(f"  变拥堵事件: {len(gt_to_congestion)} 个")
    print(f"  变畅通事件: {len(gt_to_free_flow)} 个")
    
    for config in prediction_configs:
        label = config['label']
        print(f"\n处理 {label}...")
        
        try:
            # 读取预测数据
            df_pred = pd.read_csv(config['file_path'])
            print(f"  加载数据: {len(df_pred)} 条记录")
            
            # 估计状态
            predicted_states_df = estimate_states(
                df_pred, NODE_ID,
                value_column=config['value_column'],
                window=config['window'],
                slope_threshold=config['slope_threshold'],
                lower_threshold=config['lower_threshold'],
                upper_threshold=config['upper_threshold']
            )
            
            # 限制时间范围
            predicted_states_df = predicted_states_df[
                (predicted_states_df['time'] >= start_frame) & 
                (predicted_states_df['time'] <= end_frame)
            ].copy()
            
            if len(predicted_states_df) == 0:
                print(f"  警告：未找到node_id={NODE_ID}的数据，跳过")
                continue
            
            print(f"  预测状态: {len(predicted_states_df)} 个时间点")
            
            # 对齐时间序列
            merged_df = pd.merge(
                ground_truth_states_df[['time', 'state']].rename(columns={'state': 'gt_state'}),
                predicted_states_df[['time', 'state']].rename(columns={'state': 'pred_state'}),
                on='time',
                how='inner'
            )
            
            if len(merged_df) == 0:
                print(f"  警告：无法对齐时间序列，跳过")
                continue
            
            print(f"  对齐后: {len(merged_df)} 个时间点")
            
            # 计算指标
            metrics = calculate_metrics(
                merged_df['gt_state'].values,
                merged_df['pred_state'].values
            )
            
            # 保存结果
            for state_name, state_metrics in metrics['states'].items():
                results_list.append({
                    '模型': label,
                    '状态': state_name,
                    '精确率': state_metrics['precision'],
                    '召回率': state_metrics['recall'],
                    'F1-score': state_metrics['f1'],
                    '样本数': int(state_metrics['support'])
                })
            
            # 添加加权平均
            results_list.append({
                '模型': label,
                '状态': '加权平均',
                '精确率': metrics['weighted_avg']['precision'],
                '召回率': metrics['weighted_avg']['recall'],
                'F1-score': metrics['weighted_avg']['f1'],
                '样本数': len(merged_df)
            })
            
            # 添加宏平均
            results_list.append({
                '模型': label,
                '状态': '宏平均',
                '精确率': metrics['macro_avg']['precision'],
                '召回率': metrics['macro_avg']['recall'],
                'F1-score': metrics['macro_avg']['f1'],
                '样本数': len(merged_df)
            })
            
            print(f"  加权平均 F1-score: {metrics['weighted_avg']['f1']:.4f}")
            
            # ========== 计算转换事件指标 ==========
            print(f"  检测转换事件...")
            pred_events = detect_transition_events(predicted_states_df)
            print(f"    预测转换事件: {len(pred_events)} 个")
            
            pred_to_congestion = [e for e in pred_events if e.event_type == TransitionEventType.TO_CONGESTION]
            pred_to_free_flow = [e for e in pred_events if e.event_type == TransitionEventType.TO_FREE_FLOW]
            print(f"      变拥堵事件: {len(pred_to_congestion)} 个")
            print(f"      变畅通事件: {len(pred_to_free_flow)} 个")
            
            # 计算转换事件指标
            transition_metrics = calculate_transition_metrics(
                gt_events, pred_events, TRANSITION_TIME_WINDOW
            )
            
            # 保存转换事件结果
            for event_type, metrics_dict in transition_metrics.items():
                transition_results_list.append({
                    '模型': label,
                    '事件类型': event_type,
                    '精准率': metrics_dict['precision'],
                    '召回率': metrics_dict['recall'],
                    'F1-score': metrics_dict['f1'],
                    '真正例': metrics_dict['true_positives'],
                    '假负例(漏报)': metrics_dict['false_negatives'],
                    '假正例(误报)': metrics_dict['false_positives'],
                    '平均延迟': metrics_dict['avg_delay'],
                    '延迟标准差': metrics_dict['std_delay'],
                    '延迟中位数': metrics_dict['median_delay'],
                    '绝对平均延迟': metrics_dict['avg_abs_delay'],
                    '绝对延迟中位数': metrics_dict['median_abs_delay'],
                    '真值事件数': metrics_dict['gt_count'],
                    '预测事件数': metrics_dict['pred_count']
                })
            
            print(f"    转换事件总体 F1-score: {transition_metrics['总体']['f1']:.4f}")
            print(f"    平均延迟: {transition_metrics['总体']['avg_delay']:.2f} 时间步")
            print(f"    绝对平均延迟: {transition_metrics['总体']['avg_abs_delay']:.2f} 时间步")
            
        except FileNotFoundError:
            print(f"  警告：文件 {config['file_path']} 不存在，跳过")
        except Exception as e:
            print(f"  警告：处理文件时出错: {e}")
    
    # ========== 保存结果 ==========
    if len(results_list) == 0:
        print("\n错误：没有可保存的结果！")
        return
    
    # 保存状态识别结果
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n状态识别结果已保存至: {output_path}")
    
    # 保存转换事件结果
    if len(transition_results_list) > 0:
        transition_results_df = pd.DataFrame(transition_results_list)
        transition_results_df.to_csv(transition_output_path, index=False, encoding='utf-8-sig')
        print(f"转换事件结果已保存至: {transition_output_path}")
    
    # ========== 打印摘要 ==========
    print("\n========== 状态识别结果摘要 ==========")
    for model in results_df['模型'].unique():
        model_data = results_df[results_df['模型'] == model]
        weighted_avg = model_data[model_data['状态'] == '加权平均']
        if len(weighted_avg) > 0:
            print(f"{model}:")
            print(f"  精确率: {weighted_avg['精确率'].values[0]:.4f}")
            print(f"  召回率: {weighted_avg['召回率'].values[0]:.4f}")
            print(f"  F1-score: {weighted_avg['F1-score'].values[0]:.4f}")
    
    if len(transition_results_list) > 0:
        print("\n========== 转换事件识别结果摘要 ==========")
        print(f"匹配时间窗口: {TRANSITION_TIME_WINDOW} 时间步")
        for model in transition_results_df['模型'].unique():
            model_data = transition_results_df[transition_results_df['模型'] == model]
            overall = model_data[model_data['事件类型'] == '总体']
            if len(overall) > 0:
                print(f"\n{model}:")
                print(f"  精准率: {overall['精准率'].values[0]:.4f}")
                print(f"  召回率: {overall['召回率'].values[0]:.4f}")
                print(f"  F1-score: {overall['F1-score'].values[0]:.4f}")
                print(f"  平均延迟: {overall['平均延迟'].values[0]:.2f} 时间步")
                print(f"  绝对平均延迟: {overall['绝对平均延迟'].values[0]:.2f} 时间步")
                print(f"  延迟中位数: {overall['延迟中位数'].values[0]:.2f} 时间步")
                print(f"  绝对延迟中位数: {overall['绝对延迟中位数'].values[0]:.2f} 时间步")
                
                # 显示各事件类型的详细信息
                to_congestion = model_data[model_data['事件类型'] == '变拥堵']
                to_free_flow = model_data[model_data['事件类型'] == '变畅通']
                
                if len(to_congestion) > 0:
                    print(f"  变拥堵事件:")
                    print(f"    精准率: {to_congestion['精准率'].values[0]:.4f}, 召回率: {to_congestion['召回率'].values[0]:.4f}")
                    print(f"    平均延迟: {to_congestion['平均延迟'].values[0]:.2f} 时间步")
                    print(f"    绝对平均延迟: {to_congestion['绝对平均延迟'].values[0]:.2f} 时间步")
                
                if len(to_free_flow) > 0:
                    print(f"  变畅通事件:")
                    print(f"    精准率: {to_free_flow['精准率'].values[0]:.4f}, 召回率: {to_free_flow['召回率'].values[0]:.4f}")
                    print(f"    平均延迟: {to_free_flow['平均延迟'].values[0]:.2f} 时间步")
                    print(f"    绝对平均延迟: {to_free_flow['绝对平均延迟'].values[0]:.2f} 时间步")
    
    print("\n完成!")

if __name__ == '__main__':
    main()
