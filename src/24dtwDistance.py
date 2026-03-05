'''
    输入：
        多个CSV文件，每个文件包含节点的时间序列数据，代表：真值、本模型的预测值、其他基准模型的预测值

    思路：
        从多个CSV文件中读取指定node_id的数据
        按照时间顺序排列，计算各个模型预测序列与真值序列之间的DTW距离
        DTW（Dynamic Time Warping，动态时间规整）能够衡量两个时间序列之间的相似度
        距离越小表示两个序列越相似

    关于DTW距离为inf的问题：
        如果出现DTW距离为inf，通常是因为：
        1. 窗口约束太严格，导致无法找到有效的对齐路径
        2. 序列中存在NaN值（代码会自动处理）
        3. 序列长度差异太大
        
        解决方案：
        - 将DTW_WINDOW设置为None（无约束）
        - 或者增大DTW_WINDOW的值（建议为序列长度的20-50%）
        - 检查数据质量，确保没有异常值

    输出：
        一个CSV文件，包含各模型与真值之间的DTW距离统计
'''

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = None) -> Tuple[float, np.ndarray]:
    """计算两个序列之间的DTW距离
    
    参数:
        seq1: 第一个序列（通常是真值）
        seq2: 第二个序列（通常是预测值）
        window: Sakoe-Chiba带宽约束（可选），限制匹配路径的搜索范围
    
    返回:
        (DTW距离, 累积成本矩阵)
    """
    # 检查输入是否有效
    if len(seq1) == 0 or len(seq2) == 0:
        raise ValueError("输入序列不能为空")
    
    # 检查是否有NaN值
    if np.any(np.isnan(seq1)) or np.any(np.isnan(seq2)):
        # 移除NaN值
        seq1 = seq1[~np.isnan(seq1)]
        seq2 = seq2[~np.isnan(seq2)]
        print(f"  警告：检测到NaN值，已移除。剩余长度: seq1={len(seq1)}, seq2={len(seq2)}")
    
    n, m = len(seq1), len(seq2)
    
    if n == 0 or m == 0:
        raise ValueError("移除NaN后序列为空")
    
    # 初始化累积成本矩阵
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # 如果指定了窗口约束
    if window is not None:
        window = max(window, abs(n - m))  # 窗口至少要能容纳长度差异
    
    # 动态规划填充矩阵
    for i in range(1, n + 1):
        # 确定j的搜索范围（应用Sakoe-Chiba带约束）
        if window is not None:
            j_start = max(1, i - window)
            j_end = min(m, i + window)
        else:
            j_start = 1
            j_end = m
        
        for j in range(j_start, j_end + 1):
            # 计算当前点的成本（欧氏距离的平方）
            cost = (seq1[i-1] - seq2[j-1]) ** 2
            
            # 选择最小累积成本路径
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # 插入
                dtw_matrix[i, j-1],      # 删除
                dtw_matrix[i-1, j-1]     # 匹配
            )
    
    # 检查最终结果
    if np.isinf(dtw_matrix[n, m]):
        raise ValueError(f"DTW计算失败：无法找到有效路径。窗口约束可能太严格。n={n}, m={m}, window={window}")
    
    # 返回DTW距离（取平方根得到欧氏距离）
    distance = np.sqrt(dtw_matrix[n, m])
    
    return distance, dtw_matrix

def normalized_dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = None) -> float:
    """计算归一化的DTW距离
    
    参数:
        seq1: 第一个序列
        seq2: 第二个序列
        window: Sakoe-Chiba带宽约束
    
    返回:
        归一化的DTW距离（除以序列长度）
    """
    distance, _ = dtw_distance(seq1, seq2, window)
    # 归一化：除以两个序列的平均长度
    normalized_distance = distance / ((len(seq1) + len(seq2)) / 2)
    return normalized_distance

def calculate_dtw_metrics(gt_speeds: np.ndarray, pred_speeds: np.ndarray, 
                         window: int = None) -> Dict:
    """计算DTW相关指标
    
    参数:
        gt_speeds: 真值速度序列
        pred_speeds: 预测速度序列
        window: DTW窗口约束
    
    返回:
        包含各种DTW指标的字典
    """
    # 基本DTW距离
    dtw_dist, dtw_matrix = dtw_distance(gt_speeds, pred_speeds, window)
    
    # 归一化DTW距离
    normalized_dtw = normalized_dtw_distance(gt_speeds, pred_speeds, window)
    
    # 计算路径长度（用于理解对齐复杂度）
    path_length = len(gt_speeds) + len(pred_speeds)
    
    # 计算平均逐点误差（沿着DTW路径）
    # 回溯最优路径
    path = _backtrack_path(dtw_matrix)
    path_errors = []
    for i, j in path:
        if i > 0 and j > 0:
            error = abs(gt_speeds[i-1] - pred_speeds[j-1])
            path_errors.append(error)
    
    avg_path_error = np.mean(path_errors) if path_errors else 0.0
    std_path_error = np.std(path_errors) if path_errors else 0.0
    max_path_error = np.max(path_errors) if path_errors else 0.0
    
    return {
        'dtw_distance': dtw_dist,
        'normalized_dtw': normalized_dtw,
        'path_length': len(path),
        'avg_path_error': avg_path_error,
        'std_path_error': std_path_error,
        'max_path_error': max_path_error,
        'gt_length': len(gt_speeds),
        'pred_length': len(pred_speeds)
    }

def _backtrack_path(dtw_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """回溯DTW最优路径
    
    参数:
        dtw_matrix: DTW累积成本矩阵
    
    返回:
        路径坐标列表 [(i, j), ...]
    """
    path = []
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    
    while i > 0 or j > 0:
        path.append((i, j))
        
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # 选择成本最小的前驱
            candidates = [
                (i-1, j, dtw_matrix[i-1, j]),
                (i, j-1, dtw_matrix[i, j-1]),
                (i-1, j-1, dtw_matrix[i-1, j-1])
            ]
            min_candidate = min(candidates, key=lambda x: x[2])
            i, j = min_candidate[0], min_candidate[1]
    
    path.append((0, 0))
    path.reverse()
    
    return path

def align_time_series(df1: pd.DataFrame, df2: pd.DataFrame, 
                     time_col: str = 'time', value_col1: str = 'avg_speed',
                     value_col2: str = 'avg_speed') -> Tuple[np.ndarray, np.ndarray]:
    """对齐两个时间序列（基于共同的时间点）
    
    参数:
        df1: 第一个DataFrame（真值）
        df2: 第二个DataFrame（预测）
        time_col: 时间列名
        value_col1: df1的数值列名
        value_col2: df2的数值列名
    
    返回:
        (对齐后的序列1, 对齐后的序列2)
    """
    # 合并两个DataFrame，只保留共同的时间点
    merged = pd.merge(
        df1[[time_col, value_col1]].rename(columns={value_col1: 'value1'}),
        df2[[time_col, value_col2]].rename(columns={value_col2: 'value2'}),
        on=time_col,
        how='inner'
    )
    
    # 按时间排序
    merged = merged.sort_values(time_col)
    
    # 移除包含NaN的行
    merged = merged.dropna()
    
    return merged['value1'].values, merged['value2'].values

def main():
    # ========== 配置参数 ==========
    NODE_ID = 4
    start_frame = 250
    end_frame = 600
    output_path = f'../data/draw/d210191000/vs/node{NODE_ID}_dtw_metrics.csv'
    
    # DTW窗口约束（None表示不使用约束，建议设置为序列长度的10-20%）
    # 如果出现inf错误，可以尝试设置为None或增大窗口值
    DTW_WINDOW = None  # 可调整参数，None表示无约束
    
    # ========== 数据配置 ==========
    # 真值配置
    ground_truth_config = {
        'file_path': '../data/draw/d210191000/d210291000_lane_node_stats.csv',
        'label': '真值',
        'value_column': 'avg_speed'
    }
    
    # 预测模型配置
    prediction_configs = [
        {
            'file_path': '../data/draw/d210191000/melt/0302l3/inference_results_L3.csv',
            'label': '本研究模型',
            'value_column': 'avg_speed'
        },
        {
            'file_path': '../data/draw/d210191000/vs/simple_stgnn/simple_stgnn_predictions_class10.csv',
            'label': 'STGNN',
            'value_column': 'avg_speed'
        },
        {
            'file_path': '../data/draw/d210191000/vs/physical_prior/physical_prior_predictions.csv',
            'label': '物理模型法',
            'value_column': 'avg_speed'
        }
    ]
    
    # ========== 加载真值数据 ==========
    print("加载真值数据...")
    try:
        df_gt = pd.read_csv(ground_truth_config['file_path'])
        print(f"真值数据: {len(df_gt)} 条记录")
        
        # 筛选节点和时间范围
        gt_data = df_gt[df_gt['node_id'] == NODE_ID].copy()
        gt_data = gt_data[(gt_data['time'] >= start_frame) & 
                         (gt_data['time'] <= end_frame)]
        gt_data = gt_data.sort_values('time')
        
        if len(gt_data) == 0:
            print(f"错误：未找到node_id={NODE_ID}的真值数据！")
            return
        
        print(f"真值数据: {len(gt_data)} 个时间点")
        print(f"  时间范围: {gt_data['time'].min()} - {gt_data['time'].max()}")
        print(f"  速度范围: {gt_data[ground_truth_config['value_column']].min():.2f} - "
              f"{gt_data[ground_truth_config['value_column']].max():.2f}")
        
    except FileNotFoundError:
        print(f"错误：真值文件 {ground_truth_config['file_path']} 不存在！")
        return
    except Exception as e:
        print(f"错误：处理真值文件时出错: {e}")
        return
    
    # ========== 计算各模型的DTW距离 ==========
    print(f"\n计算DTW距离（窗口约束: {DTW_WINDOW if DTW_WINDOW else '无'}）...")
    results_list = []
    
    for config in prediction_configs:
        label = config['label']
        print(f"\n处理 {label}...")
        
        try:
            # 读取预测数据
            df_pred = pd.read_csv(config['file_path'])
            print(f"  加载数据: {len(df_pred)} 条记录")
            
            # 筛选节点和时间范围
            pred_data = df_pred[df_pred['node_id'] == NODE_ID].copy()
            pred_data = pred_data[(pred_data['time'] >= start_frame) & 
                                 (pred_data['time'] <= end_frame)]
            pred_data = pred_data.sort_values('time')
            
            if len(pred_data) == 0:
                print(f"  警告：未找到node_id={NODE_ID}的数据，跳过")
                continue
            
            print(f"  预测数据: {len(pred_data)} 个时间点")
            print(f"    时间范围: {pred_data['time'].min()} - {pred_data['time'].max()}")
            print(f"    速度范围: {pred_data[config['value_column']].min():.2f} - "
                  f"{pred_data[config['value_column']].max():.2f}")
            
            # 对齐时间序列
            gt_speeds, pred_speeds = align_time_series(
                gt_data, pred_data,
                time_col='time',
                value_col1=ground_truth_config['value_column'],
                value_col2=config['value_column']
            )
            
            if len(gt_speeds) == 0:
                print(f"  警告：无法对齐时间序列，跳过")
                continue
            
            print(f"  对齐后: {len(gt_speeds)} 个时间点")
            
            # 计算DTW指标
            print(f"  计算DTW距离...")
            try:
                metrics = calculate_dtw_metrics(gt_speeds, pred_speeds, window=DTW_WINDOW)
            except ValueError as e:
                print(f"  警告：使用窗口约束{DTW_WINDOW}计算失败: {e}")
                print(f"  尝试使用无约束模式...")
                try:
                    metrics = calculate_dtw_metrics(gt_speeds, pred_speeds, window=None)
                    print(f"  无约束模式计算成功")
                except Exception as e2:
                    print(f"  错误：无约束模式也失败: {e2}")
                    continue
            
            # 保存结果
            results_list.append({
                '模型': label,
                'DTW距离': metrics['dtw_distance'],
                '归一化DTW距离': metrics['normalized_dtw'],
                '路径长度': metrics['path_length'],
                '平均路径误差': metrics['avg_path_error'],
                '路径误差标准差': metrics['std_path_error'],
                '最大路径误差': metrics['max_path_error'],
                '真值序列长度': metrics['gt_length'],
                '预测序列长度': metrics['pred_length'],
                '窗口约束': DTW_WINDOW if DTW_WINDOW else '无'
            })
            
            print(f"  DTW距离: {metrics['dtw_distance']:.4f}")
            print(f"  归一化DTW距离: {metrics['normalized_dtw']:.4f}")
            print(f"  平均路径误差: {metrics['avg_path_error']:.4f}")
            
        except FileNotFoundError:
            print(f"  警告：文件 {config['file_path']} 不存在，跳过")
        except Exception as e:
            print(f"  警告：处理文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 保存结果 ==========
    if len(results_list) == 0:
        print("\n错误：没有可保存的结果！")
        return
    
    results_df = pd.DataFrame(results_list)
    
    # 按DTW距离排序（距离越小越好）
    results_df = results_df.sort_values('DTW距离')
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nDTW距离结果已保存至: {output_path}")
    
    # ========== 打印摘要 ==========
    print("\n========== DTW距离结果摘要 ==========")
    print(f"节点ID: {NODE_ID}")
    print(f"时间范围: {start_frame} - {end_frame}")
    print(f"窗口约束: {DTW_WINDOW if DTW_WINDOW else '无'}")
    print("\n模型排名（按DTW距离从小到大）:")
    for idx, row in results_df.iterrows():
        print(f"\n{row['模型']}:")
        print(f"  DTW距离: {row['DTW距离']:.4f}")
        print(f"  归一化DTW距离: {row['归一化DTW距离']:.6f}")
        print(f"  平均路径误差: {row['平均路径误差']:.4f}")
        print(f"  路径误差标准差: {row['路径误差标准差']:.4f}")
    
    print("\n完成!")

if __name__ == '__main__':
    main()
