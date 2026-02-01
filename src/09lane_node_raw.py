# -*- coding: utf-8 -*-
"""
09lane_node.py

按照节点ID(node_id)和时间帧统计每1秒当前节点内的交通状况（使用多滑块滑动时间窗口）

多滑块窗口说明：
- 输出仍然对应每一秒（如第11秒、第12秒...）
- 不同指标使用不同大小的滑动窗口，以适应各自的时间特性：
  * 速度：1秒窗口 - 捕捉瞬时速度变化
  * 流量：10秒窗口 - 累积足够的车辆数，平滑随机波动
  * 占用率：4秒窗口 - 平衡敏感度和稳定性
- 例如：第11秒的输出中
  * 速度统计第10.5~11.5秒的数据（1秒窗口）
  * 流量统计第6~16秒的数据（10秒窗口）
  * 占用率统计第9~13秒的数据（4秒窗口）
- 输出从第 MAX_HALF_WINDOW 秒开始，到倒数第 MAX_HALF_WINDOW 秒结束
  （确保所有指标都有完整的滑动窗口数据）

输入：
- 轨迹CSV（来自05trajectory_with_laneid.py），包含 id, frame, FID(node_id), car_type, v 等字段
- graph.json（道路图结构，包含 lanes 和 nodes）

输出：
- CSV文件，每行代表一个节点在1秒内的交通状况（基于多滑块滑动窗口平均）
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict


# =================== 配置参数 ===================
# 节点段长度（米），用于计算占用率
SEGMENT_LENGTH = 10.0  # 默认10米

# 车辆类型占用长度（米）
VEHICLE_LENGTHS = {
    'car': 4.0,
    'medium': 6.0,
    'heavy': 10.0,
    'motorcycle': 2.0
}

# 滑动时间窗口大小（秒）- 为不同指标设置不同的窗口
SPEED_WINDOW = 2.0       # 速度滑块：2秒（捕捉瞬时速度变化）
FLOW_WINDOW = 10.0       # 流量滑块：10秒（累积足够的车辆数）
OCCUPANCY_WINDOW = 4.0   # 占用率滑块：4秒（平衡敏感度和稳定性）

# 计算最大窗口半径（用于确定输出时间范围）
MAX_WINDOW = max(SPEED_WINDOW, FLOW_WINDOW, OCCUPANCY_WINDOW)
MAX_HALF_WINDOW = int(MAX_WINDOW / 2)

# 各指标的窗口半径
SPEED_HALF_WINDOW = SPEED_WINDOW / 2
FLOW_HALF_WINDOW = FLOW_WINDOW / 2
OCCUPANCY_HALF_WINDOW = OCCUPANCY_WINDOW / 2


def load_graph(graph_json_path):
    """
    加载图结构，构建 node_id 到节点信息的映射
    
    参数:
        graph_json_path: str, graph.json文件路径
        
    返回:
        dict: {node_id: {'lane_id': int, 'position_in_lane': float, 'segment_length': float, 
                         'direct': set, 'near': set, 'crossing': set}}
    """
    print(f"📦 正在读取图结构: {graph_json_path}")
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 构建 node_id -> node_info 的映射
    node_dict = {}
    for node in graph_data.get('nodes', []):
        node_id = int(node['node_id'])
        connections = node.get('node_connections', {})
        node_dict[node_id] = {
            'lane_id': int(node.get('lane_id', -1)),
            'position_in_lane': node.get('position_in_lane'),
            'segment_length': node.get('segment_length', SEGMENT_LENGTH),
            'direct': set(connections.get('direct', [])),
            'near': set(connections.get('near', [])),
            'crossing': set(connections.get('crossing', []))
        }
    
    print(f"✅ 共加载 {len(node_dict)} 个节点")
    return node_dict


def get_vehicle_length(car_type):
    """
    根据车辆类型获取占用长度
    
    参数:
        car_type: str, 车辆类型
        
    返回:
        float: 占用长度（米）
    """
    if pd.isna(car_type) or car_type is None:
        # 如果车辆类型未知，使用car的默认值
        return VEHICLE_LENGTHS.get('car', 4.0)
    
    car_type_str = str(car_type).lower().strip()
    return VEHICLE_LENGTHS.get(car_type_str, VEHICLE_LENGTHS.get('car', 4.0))


def get_next_node_for_vehicle(traj_df, vehicle_id, current_node_id, current_frame):
    """
    获取车辆在当前节点之后下一个经过的节点
    
    参数:
        traj_df: DataFrame, 轨迹数据
        vehicle_id: 车辆ID
        current_node_id: 当前节点ID
        current_frame: 当前时间帧
        
    返回:
        int or None: 下一个节点ID，如果没有则返回None
    """
    # 获取该车辆的所有轨迹点，按frame排序
    vehicle_traj = traj_df[traj_df['id'] == vehicle_id].sort_values('frame')
    
    # 找到当前frame之后的所有轨迹点
    future_traj = vehicle_traj[vehicle_traj['frame'] > current_frame]
    
    if future_traj.empty:
        return None
    
    # 找到第一个与当前节点不同的节点
    current_node_str = str(current_node_id)
    for _, row in future_traj.iterrows():
        next_node_str = str(row['FID'])
        if next_node_str != current_node_str and pd.notna(row['FID']):
            try:
                return int(float(next_node_str))
            except (ValueError, TypeError):
                return None
    
    return None


def classify_trajectory_type(current_node_id, next_node_id, node_dict):
    """
    根据当前节点和下一个节点判断轨迹类型
    
    参数:
        current_node_id: int, 当前节点ID
        next_node_id: int or None, 下一个节点ID
        node_dict: dict, 节点字典
        
    返回:
        str or None: 'crossing', 'direct', 'near' 或 None
    """
    if next_node_id is None:
        return None
    
    # 获取当前节点的连接信息
    if current_node_id not in node_dict:
        return None
    
    node_info = node_dict[current_node_id]
    
    # 检查是否属于crossing
    if next_node_id in node_info['crossing']:
        return 'crossing'
    
    # 检查是否属于direct
    if next_node_id in node_info['direct']:
        return 'direct'
    
    # 检查是否属于near
    if next_node_id in node_info['near']:
        return 'near'
    
    return None


def calculate_occupancy_rate(group, segment_length):
    """
    计算占用率
    
    参数:
        group: DataFrame, 某一帧内的所有车辆记录
        segment_length: float, 节点段长度（米）
        
    返回:
        float: 占用率（0-1之间）
    """
    if group.empty:
        return 0.0
    
    # 计算所有车辆的占用长度之和
    total_length = 0.0
    for _, row in group.iterrows():
        car_type = row.get('car_type')
        vehicle_length = get_vehicle_length(car_type)
        total_length += vehicle_length
    
    # 占用率 = 总占用长度 / 节点段长度
    occupancy_rate = min(total_length / segment_length, 1.0)  # 限制在0-1之间
    
    return occupancy_rate


def main(traj_csv_path, graph_json_path, output_csv_path):
    """
    主函数
    
    参数:
        traj_csv_path: str, 轨迹CSV路径
        graph_json_path: str, graph.json文件路径
        output_csv_path: str, 输出CSV路径
    """
    print("🚀 开始统计节点交通状况...")
    
    # =================== Step 1: 读取数据 ===================
    print("📦 正在读取轨迹数据...")
    traj_df = pd.read_csv(traj_csv_path)
    
    # 检查必要字段
    required_fields = ['id', 'frame', 'FID', 'v']
    missing_fields = [f for f in required_fields if f not in traj_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 轨迹数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(traj_df)} 条轨迹记录")
    
    # 处理frame字段
    if 'frame' in traj_df.columns:
        traj_df['frame'] = traj_df['frame'].astype(str).str.rstrip(';')
        traj_df['frame'] = traj_df['frame'].astype(float)
    
    # 过滤掉没有节点ID的记录
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna()].copy()
    traj_df = traj_df[traj_df['FID'].astype(str).str.strip() != ''].copy()
    filtered_count = len(traj_df)
    print(f"📊 过滤后保留 {filtered_count} 条有效记录（过滤前: {original_count}）")
    
    # 确保FID为字符串类型
    traj_df['FID'] = traj_df['FID'].astype(str).str.strip()
    
    # 加载图结构
    node_dict = load_graph(graph_json_path)
    
    # =================== Step 2: 确定时间窗口 ===================
    print("🕐 正在确定时间窗口...")
    min_frame = traj_df['frame'].min()
    max_frame = traj_df['frame'].max()
    
    # 使用最大窗口半径来确定输出范围，确保所有指标都有完整的滑动窗口数据
    output_start = min_frame + MAX_HALF_WINDOW
    output_end = max_frame - MAX_HALF_WINDOW
    
    # 生成输出时间点（每1秒一个）
    output_times = []
    current_time = output_start
    while current_time <= output_end:
        output_times.append(current_time)
        current_time += 1
    
    print(f"✅ 原始数据范围: {min_frame:.2f} ~ {max_frame:.2f}")
    print(f"✅ 速度滑动窗口: {SPEED_WINDOW} 秒")
    print(f"✅ 流量滑动窗口: {FLOW_WINDOW} 秒")
    print(f"✅ 占用率滑动窗口: {OCCUPANCY_WINDOW} 秒")
    print(f"✅ 输出时间范围: {output_start:.2f} ~ {output_end:.2f}")
    print(f"✅ 共生成 {len(output_times)} 个输出时间点")
    
    # =================== Step 3: 按节点和时间窗口统计 ===================
    print("📊 正在统计每个节点在每个滑动时间窗口的交通状况...")
    
    results = []
    
    # 获取所有节点ID
    all_node_ids = set(node_dict.keys())
    
    # 将FID转换为整数，方便匹配
    traj_df['FID_int'] = traj_df['FID'].apply(lambda x: int(float(x)) if x else -1)
    
    # 对每个节点和每个输出时间点进行统计
    for node_id in all_node_ids:
        node_group = traj_df[traj_df['FID_int'] == node_id]
        
        # 获取该节点的段长度
        node_info = node_dict.get(node_id, {})
        segment_length = node_info.get('segment_length', SEGMENT_LENGTH)
        
        # 对该节点的每个输出时间点进行统计
        for output_time in output_times:
            # ========== 1. 计算速度（使用1秒窗口）==========
            speed_window_start = output_time - SPEED_HALF_WINDOW
            speed_window_end = output_time + SPEED_HALF_WINDOW
            speed_window_data = node_group[
                (node_group['frame'] >= speed_window_start) & 
                (node_group['frame'] < speed_window_end)
            ]
            
            # 计算平均速度（绝对值，单位：km/h）
            if speed_window_data.empty:
                avg_speed = None
            else:
                avg_speed = speed_window_data['v'].abs().mean()
                if pd.isna(avg_speed):
                    avg_speed = None
                else:
                    avg_speed = round(avg_speed, 2)
            
            # ========== 2. 计算流量（使用10秒窗口）==========
            flow_window_start = output_time - FLOW_HALF_WINDOW
            flow_window_end = output_time + FLOW_HALF_WINDOW
            flow_window_data = node_group[
                (node_group['frame'] >= flow_window_start) & 
                (node_group['frame'] < flow_window_end)
            ]
            
            # 统计唯一车辆数
            unique_vehicles = flow_window_data['id'].nunique() if not flow_window_data.empty else 0
            
            # ========== 3. 计算占用率（使用4秒窗口）==========
            occupancy_window_start = output_time - OCCUPANCY_HALF_WINDOW
            occupancy_window_end = output_time + OCCUPANCY_HALF_WINDOW
            occupancy_window_data = node_group[
                (node_group['frame'] >= occupancy_window_start) & 
                (node_group['frame'] < occupancy_window_end)
            ]
            
            # 计算平均占用率（需要统计每一帧的占用率，然后求平均）
            if occupancy_window_data.empty:
                avg_occupancy = 0.0
            else:
                frame_occupancies = []
                for frame, frame_group in occupancy_window_data.groupby('frame'):
                    occupancy = calculate_occupancy_rate(frame_group, segment_length)
                    frame_occupancies.append(occupancy)
                avg_occupancy = np.mean(frame_occupancies) if frame_occupancies else 0.0
            
            # 保存结果
            results.append({
                'node_id': node_id,
                'start_frame': output_time,  # 输出时间点（滑动窗口中心）
                'avg_speed': avg_speed,
                'avg_occupancy': round(avg_occupancy, 2),
                'total_vehicles': unique_vehicles,
            })
    
    # =================== Step 4: 保存结果 ===================
    print(f"💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['node_id', 'start_frame']).reset_index(drop=True)
    
    # =================== 归一化处理 ===================
    # avg_speed: 不进行归一化，保持原始值（km/h），空值保持为空值
    
    # total_vehicles: 按对数变换 + 归一化
    results_df['total_vehicles'] = results_df['total_vehicles'].apply(
        lambda x: round(np.log(1 + x) / np.log(15) , 2)
    )
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 统计结果已保存至: {output_csv_path}")
    print(f"📊 总计统计记录数: {len(results_df)}")
    print(f"📊 涉及节点数: {results_df['node_id'].nunique()}")
    print(f"📊 时间窗口数: {results_df['start_frame'].nunique()}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    TRAJ_CSV_PATH = r"../data/trajectory_with_laneid/d210291000.csv"  # 轨迹数据
    GRAPH_JSON_PATH = r"../data/road_graph/graph_10m.json"  # 图结构（更新版本，包含lanes和nodes）
    OUTPUT_CSV = r"../data/lane_node_stats/d210291000_lane_node_stats.csv"  # 输出路径
    
    if not os.path.exists(GRAPH_JSON_PATH):
        raise FileNotFoundError(f"❌ 图文件不存在: {GRAPH_JSON_PATH}")
    
    # 执行统计
    main(TRAJ_CSV_PATH, GRAPH_JSON_PATH, OUTPUT_CSV)

