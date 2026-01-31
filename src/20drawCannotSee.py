'''
    输入：
        完整的各节点各时间步真值：data\draw\input\d210191000\d210291000_lane_node_stats.csv
        可见轮巡窗口：data\lane_node_stats\d210291000_node_mask.csv

    处理目标：
        绘制符合期刊论文要求的图表，可见的部分用实线、不可见的用虚线

    思路：
        先从d210291000_lane_node_stats.csv中筛选出node_id等于1的行，下面所有处理和显示的图表都局限于node_id等于1
        然后增加一列is_observed，对每一行基于node_id和time，对是否在可见轮巡窗口进行判断，是则为1，不是则为0
        接着绘制随着时间avg_speed变化的折线，对于is_observed为1的部分用实线连接，为0的用虚线连接

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
        start_time = row['relative_start']
        end_time = row['relative_end']
        
        # 检查时间是否在可见窗口内
        if start_time <= time <= end_time:
            return True
    
    return False

def plot_speed_with_visibility(df, output_path, dpi=300):
    """绘制速度随时间变化的图表，区分可见和不可见部分"""
    
    # 设置图表样式
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = df['time'].values
    speeds = df['avg_speed'].values
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
    
    # 绘制每个段
    for segment in segments:
        if segment['observed']:
            # 可见部分：蓝色实线
            ax.plot(segment['times'], segment['speeds'], 'b-', linewidth=2, label='Observed' if 'Observed' not in ax.get_legend_handles_labels()[1] else '')
        else:
            # 不可见部分：红色虚线
            ax.plot(segment['times'], segment['speeds'], 'r--', linewidth=2, label='Unobserved' if 'Unobserved' not in ax.get_legend_handles_labels()[1] else '')
    
    # 设置图表属性
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Average Speed (km/h)', fontsize=12)
    ax.set_title('Speed Variation with Observation Status (Node 1)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    plt.close()

def main():
    # 文件路径
    lane_stats_path = 'data/draw/input/d210191000/d210291000_lane_node_stats.csv'
    node_mask_path = 'data/lane_node_stats/d210291000_node_mask.csv'
    output_path = 'plots/node1_speed_visibility.png'
    
    # 1. 加载数据
    print("加载数据...")
    lane_stats = pd.read_csv(lane_stats_path)
    node_mask = pd.read_csv(node_mask_path)
    
    print(f"车道统计数据: {len(lane_stats)} 条记录")
    print(f"节点可见窗口: {len(node_mask)} 条记录")
    
    # 2. 筛选node_id=1的数据
    print("\n筛选node_id=1的数据...")
    node1_data = lane_stats[lane_stats['node_id'] == 1].copy()
    print(f"找到 {len(node1_data)} 条记录")
    
    # 打印node_id=1的可见窗口
    node1_windows = node_mask[node_mask['node_id'] == 1]
    print(f"\nNode 1 的可见窗口数量: {len(node1_windows)}")
    if len(node1_windows) > 0:
        print("前5个窗口:")
        print(node1_windows.head())
    
    # 3. 添加is_observed列
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
    plot_speed_with_visibility(node1_data, output_path)
    
    print("\n完成!")

if __name__ == '__main__':
    main()
