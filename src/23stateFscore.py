'''
    输入：
        多个CSV文件，每个文件包含节点的时间序列数据，代表：真值、本模型的预测值、其他基准模型的预测值

    思路：
        从多个CSV文件中读取指定node_id的数据
        按照时间顺序排列，对每个时间节点所处的交通状态进行判断
        将各个模型的预测状态与真值状态进行对比
        计算精确率(Precision)、召回率(Recall)、F1-score

    输出：
        一个CSV文件，包含各个模型的精确率、召回率、F1-score统计
'''

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

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
    NODE_ID = 20
    start_frame = 250
    end_frame = 600
    output_path = f'../data/draw/d210191000/vs/node{NODE_ID}_fscore_metrics.csv'
    
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
            'file_path': '../data/draw/d210191000/vs/simple_stgnn/simple_stgnn_predictions_class10.csv',
            'label': 'STGNN',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 18
        },
        {
            'file_path': '../data/draw/d210191000/vs/physical_prior/physical_prior_predictions.csv',
            'label': '物理模型法',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 15
        },
        {
            'file_path': '../data/draw/d210191000/vs/phase_template/phase_template_results.csv',
            'label': '模板法',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 20
        },
        {
            'file_path': '../data/draw/d210191000/vs/st_idw/st_idw_results.csv',
            'label': 'ST-IDW',
            'value_column': 'avg_speed',
            'window': 5,
            'slope_threshold': 0.155,
            'lower_threshold': 5,
            'upper_threshold': 20
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
            
        except FileNotFoundError:
            print(f"  警告：文件 {config['file_path']} 不存在，跳过")
        except Exception as e:
            print(f"  警告：处理文件时出错: {e}")
    
    # ========== 保存结果 ==========
    if len(results_list) == 0:
        print("\n错误：没有可保存的结果！")
        return
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")
    
    # ========== 打印摘要 ==========
    print("\n========== 结果摘要 ==========")
    for model in results_df['模型'].unique():
        model_data = results_df[results_df['模型'] == model]
        weighted_avg = model_data[model_data['状态'] == '加权平均']
        if len(weighted_avg) > 0:
            print(f"{model}:")
            print(f"  精确率: {weighted_avg['精确率'].values[0]:.4f}")
            print(f"  召回率: {weighted_avg['召回率'].values[0]:.4f}")
            print(f"  F1-score: {weighted_avg['F1-score'].values[0]:.4f}")
    
    print("\n完成!")

if __name__ == '__main__':
    main()
