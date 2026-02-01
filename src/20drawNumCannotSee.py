'''
    输入：
        完整的各节点各时间步真值：data\draw\input\d210191000\d210291000_lane_node_stats.csv
        可见轮巡窗口：data\lane_node_stats\d210291000_node_mask.csv

    处理目标：
        绘制符合期刊论文要求的图表，可见的部分用实线、不可见的用虚线

    思路：
        先从d210291000_lane_node_stats.csv中筛选出node_id等于1的行，下面所有处理和显示的图表都局限于node_id等于1
        然后增加一列is_observed，对每一行基于node_id和time，对是否在可见轮巡窗口进行判断，是则为1，不是则为0
        接着绘制随着时间avg_occupancy变化的折线，对于is_observed为1的部分用实线连接，为0的用虚线连接

    输出：
        一张300pi图表png
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_node_visibility(node_id, time, node_mask):
    """检查某个节点在某个时间是否可见"""
    # 筛选该节点的可见窗口
    node_windows = node_mask[node_mask['node_id'] == node_id]
    
    for _, row in node_windows.iterrows():
        start_time = row['start']
        end_time = row['end']
        
        # 检查时间是否在可见窗口内
        if start_time <= time <= end_time:
            return True
    
    return False

def plot_speed_with_visibility(df, interpolated_df, additional_df, output_path, start_frame=5, end_frame=824, dpi=300,
                               show_interpolated=True, show_additional=True, show_observed=True, show_unobserved=True):
    """绘制流量随时间变化的图表，区分可见和不可见部分，并添加插值数据和额外数据
    
    参数:
        show_interpolated: 是否显示插值数据（灰色虚线）
        show_additional: 是否显示额外数据（绿色实线）
        show_observed: 是否显示可见部分（蓝色点）
        show_unobserved: 是否显示不可见部分（红色虚线）
    """
    
    # 设置图表样式 - 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 限制时间范围并按时间排序
    df = df[(df['time'] >= start_frame) & (df['time'] <= end_frame)].copy()
    df = df.sort_values('time')
    interpolated_df = interpolated_df[(interpolated_df['time'] >= start_frame) & (interpolated_df['time'] <= end_frame)].copy()
    interpolated_df = interpolated_df.sort_values('time')
    additional_df = additional_df[(additional_df['time'] >= start_frame) & (additional_df['time'] <= end_frame)].copy()
    additional_df = additional_df.sort_values('time')
    
    # 首先绘制插值数据（灰色虚线，在最下层）
    if show_interpolated and len(interpolated_df) > 0:
        print(f"绘制插值数据: {len(interpolated_df)} 个点")
        print(f"插值数据时间范围: {interpolated_df['time'].min()} - {interpolated_df['time'].max()}")
        print(f"插值流量范围: {interpolated_df['avg_occupancy'].min():.2f} - {interpolated_df['avg_occupancy'].max():.2f}")
        ax.plot(interpolated_df['time'].values, interpolated_df['avg_occupancy'].values, 
                color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                label='线性插值结果', zorder=1)
    elif not show_interpolated:
        print("插值数据显示已关闭")
    else:
        print("警告：插值数据为空，跳过绘制")
    
    # 绘制额外数据（绿色实线）
    if show_additional and len(additional_df) > 0:
        ax.plot(additional_df['time'].values, additional_df['avg_occupancy'].values, 
                color='green', linestyle='-', linewidth=1.5, alpha=0.7, 
                label='模型补全结果', zorder=1.5)
    elif not show_additional:
        print("额外数据显示已关闭")
    
    times = df['time'].values
    speeds = df['avg_occupancy'].values
    is_observed = df['is_observed'].values
    
    # 找出所有可见和不可见的连续段
    segments = []
    current_segment = {'start': 0, 'observed': is_observed[0], 'times': [times[0]], 'speeds': [speeds[0]]}
    
    for i in range(1, len(times)):
        if is_observed[i] == current_segment['observed']:
            current_segment['times'].append(times[i])
            current_segment['speeds'].append(speeds[i])
        else:
            segments.append(current_segment)
            current_segment = {'start': i, 'observed': is_observed[i], 'times': [times[i]], 'speeds': [speeds[i]]}
    
    segments.append(current_segment)
    
    # 绘制每个段（在插值数据之上）
    for segment in segments:
        if segment['observed'] and show_observed:
            # 可见部分：蓝色点
            ax.scatter(segment['times'], segment['speeds'], c='blue', s=5, marker='o',
                      label='可见段' if '可见段' not in ax.get_legend_handles_labels()[1] else '',
                      zorder=3)
        elif not segment['observed'] and show_unobserved:
            # 不可见部分：红色虚线
            ax.plot(segment['times'], segment['speeds'], 'r--', linewidth=1, 
                   label='不可见段' if '不可见段' not in ax.get_legend_handles_labels()[1] else '',
                   zorder=2)
    
    # 设置图表属性
    ax.set_xlabel('相对时间(s)', fontsize=12)
    ax.set_ylabel('实时流量', fontsize=12)
    ax.set_title(f'可见状态下的流量特征变化曲线 (节点{df["node_id"].iloc[0] if len(df) > 0 else "N/A"})', fontsize=14, fontweight='bold')
    ax.set_xlim(start_frame, end_frame)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    plt.close()

def main():
    # ========== 配置参数 ==========
    # 节点ID（修改此处以分析不同节点）
    NODE_ID = 42
    
    # 显示选项（True=显示，False=隐藏）
    SHOW_INTERPOLATED = True    # 插值数据（灰色虚线）
    SHOW_ADDITIONAL = True       # 额外数据（绿色实线）
    SHOW_OBSERVED = True         # 可见部分（蓝色点）
    SHOW_UNOBSERVED = True       # 不可见部分（红色虚线）
    
    # 文件路径
    lane_stats_path = '../data/draw/d210191000/d210291000_lane_node_stats.csv'
    node_mask_path = '../data/draw/d210191000/d210291000_node_mask.csv'
    interpolated_data_path = '../data/draw/d210191000/interpolated_data.csv'
    additional_data_path = '../data/draw/d210191000/inference_results.csv'
    
    output_path = f'../data/draw/d210191000/node{NODE_ID}_num_visibility.png'

    start_frame = 5
    end_frame = 824
    
    # 1. 加载数据
    print("加载数据...")
    lane_stats = pd.read_csv(lane_stats_path)
    node_mask = pd.read_csv(node_mask_path)
    interpolated_data = pd.read_csv(interpolated_data_path)
    additional_data = pd.read_csv(additional_data_path)
    
    print(f"车道统计数据: {len(lane_stats)} 条记录")
    print(f"节点可见窗口: {len(node_mask)} 条记录")
    print(f"插值数据: {len(interpolated_data)} 条记录")
    print(f"额外数据: {len(additional_data)} 条记录")
    
    # 2. 筛选指定node_id的数据
    print(f"\n筛选node_id={NODE_ID}的数据...")
    node1_data = lane_stats[lane_stats['node_id'] == NODE_ID].copy()
    # 筛选指定node_id的插值数据
    node1_interpolated = interpolated_data[interpolated_data['node_id'] == NODE_ID].copy()
    # 筛选额外数据中指定node_id的数据
    node1_additional = additional_data[additional_data['node_id'] == NODE_ID].copy()
    
    print(f"真值数据找到 {len(node1_data)} 条记录")
    print(f"插值数据找到 {len(node1_interpolated)} 条记录")
    print(f"额外数据找到 {len(node1_additional)} 条记录")
    
    # 将 interpolated_avg_occupancy 转换为 avg_occupancy（乘以100）
    if len(node1_interpolated) > 0:
        print(f"\n插值数据列名: {node1_interpolated.columns.tolist()}")
        if 'interpolated_avg_occupancy' in node1_interpolated.columns:
            node1_interpolated['avg_occupancy'] = node1_interpolated['interpolated_avg_occupancy']
            print(f"插值数据 avg_occupancy 范围: {node1_interpolated['avg_occupancy'].min():.2f} - {node1_interpolated['avg_occupancy'].max():.2f}")
        elif 'avg_occupancy' not in node1_interpolated.columns:
            print("警告：插值数据中既没有 'interpolated_avg_occupancy' 也没有 'avg_occupancy' 列！")
        else:
            print("插值数据已包含 'avg_occupancy' 列，无需转换")
    else:
        print("警告：没有找到符合条件的插值数据！")
    
    # 检查插值数据的时间连续性
    if len(node1_interpolated) > 0:
        node1_interpolated_sorted = node1_interpolated.sort_values('time')
        time_diffs = node1_interpolated_sorted['time'].diff()
        non_consecutive = time_diffs[time_diffs > 1]
        if len(non_consecutive) > 0:
            print(f"警告：插值数据存在时间间隔 > 1 的情况，共 {len(non_consecutive)} 处")
            print(f"时间范围: {node1_interpolated_sorted['time'].min()} - {node1_interpolated_sorted['time'].max()}")
        else:
            print(f"插值数据时间连续，范围: {node1_interpolated_sorted['time'].min()} - {node1_interpolated_sorted['time'].max()}")
    
    # 打印指定node_id的可见窗口
    node1_windows = node_mask[node_mask['node_id'] == NODE_ID]
    print(f"\nNode {NODE_ID} 的可见窗口数量: {len(node1_windows)}")
    if len(node1_windows) > 0:
        print("前5个窗口:")
        print(node1_windows.head())
    
    # 3. 添加is_observed列（仅对真值数据）
    print("\n判断可见性...")
    node1_data['is_observed'] = node1_data.apply(
        lambda row: check_node_visibility(row['node_id'], row['time'], node_mask),
        axis=1
    )
    
    # 转换为整数
    node1_data['is_observed'] = node1_data['is_observed'].astype(int)
    
    # 打印统计信息
    observed_count = node1_data['is_observed'].sum()
    total_count = len(node1_data)
    print(f"可见时间步: {observed_count}/{total_count} ({observed_count/total_count*100:.1f}%)")
    
    # 4. 绘制图表
    print("\n绘制图表...")
    plot_speed_with_visibility(node1_data, node1_interpolated, node1_additional, output_path, 
                              start_frame, end_frame,
                              show_interpolated=SHOW_INTERPOLATED,
                              show_additional=SHOW_ADDITIONAL,
                              show_observed=SHOW_OBSERVED,
                              show_unobserved=SHOW_UNOBSERVED)
    
    print("\n完成!")

if __name__ == '__main__':
    main()
