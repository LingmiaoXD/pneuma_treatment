# -*- coding: utf-8 -*-
"""
09lane_node.py

按照车道段ID和时间帧统计每1秒当前车道段内的交通状况

输入：
- 轨迹CSV（来自05trajectory_with_laneid.py），包含 id, frame, FID, car_type, v 等字段
- graph.json（道路图结构）

输出：
- CSV文件，每行代表一个车道段在1秒内的交通状况
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict


# =================== 配置参数 ===================
# 路段长度（米），方便开发者调试
LANE_LENGTH = 40.0  # 默认40米

# 车辆类型占用长度（米）
VEHICLE_LENGTHS = {
    'car': 4.0,
    'medium': 8.0,
    'heavy': 14.0,
    'motorcycle': 2.0
}

# 时间窗口大小（秒）
TIME_WINDOW = 1.0


def load_graph(graph_json_path):
    """
    加载图结构
    
    参数:
        graph_json_path: str, graph.json文件路径
        
    返回:
        dict: {lane_id: {'direct': [...], 'near': [...], 'crossing': [...]}}
    """
    print(f"📦 正在读取图结构: {graph_json_path}")
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 构建快速查找字典
    graph_dict = {}
    for node in graph_data.get('nodes', []):
        lane_id = int(node['lane_id'])
        connections = node.get('node_connections', {})
        graph_dict[lane_id] = {
            'direct': set(connections.get('direct', [])),
            'near': set(connections.get('near', [])),
            'crossing': set(connections.get('crossing', []))
        }
    
    print(f"✅ 共加载 {len(graph_dict)} 个车道段节点")
    return graph_dict


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


def get_next_lane_for_vehicle(traj_df, vehicle_id, current_lane_id, current_frame):
    """
    获取车辆在当前车道段之后下一个经过的车道段
    
    参数:
        traj_df: DataFrame, 轨迹数据
        vehicle_id: 车辆ID
        current_lane_id: 当前车道段ID
        current_frame: 当前时间帧
        
    返回:
        int or None: 下一个车道段ID，如果没有则返回None
    """
    # 获取该车辆的所有轨迹点，按frame排序
    vehicle_traj = traj_df[traj_df['id'] == vehicle_id].sort_values('frame')
    
    # 找到当前frame之后的所有轨迹点
    future_traj = vehicle_traj[vehicle_traj['frame'] > current_frame]
    
    if future_traj.empty:
        return None
    
    # 找到第一个与当前车道段不同的车道段
    current_lane_str = str(current_lane_id)
    for _, row in future_traj.iterrows():
        next_lane_str = str(row['FID'])
        if next_lane_str != current_lane_str and pd.notna(row['FID']):
            try:
                return int(float(next_lane_str))
            except (ValueError, TypeError):
                return None
    
    return None


def classify_trajectory_type(current_lane_id, next_lane_id, graph_dict):
    """
    根据当前车道段和下一个车道段判断轨迹类型
    
    参数:
        current_lane_id: int, 当前车道段ID
        next_lane_id: int or None, 下一个车道段ID
        graph_dict: dict, 图结构字典
        
    返回:
        str or None: 'crossing', 'direct', 'near' 或 None
    """
    if next_lane_id is None:
        return None
    
    # 获取当前车道段的连接信息
    if current_lane_id not in graph_dict:
        return None
    
    connections = graph_dict[current_lane_id]
    
    # 检查是否属于crossing
    if next_lane_id in connections['crossing']:
        return 'crossing'
    
    # 检查是否属于direct
    if next_lane_id in connections['direct']:
        return 'direct'
    
    # 检查是否属于near
    if next_lane_id in connections['near']:
        return 'near'
    
    return None


def calculate_occupancy_rate(group, lane_length):
    """
    计算占用率
    
    参数:
        group: DataFrame, 某一帧内的所有车辆记录
        lane_length: float, 路段长度（米）
        
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
    
    # 占用率 = 总占用长度 / 路段长度
    occupancy_rate = min(total_length / lane_length, 1.0)  # 限制在0-1之间
    
    return occupancy_rate


def main(traj_csv_path, graph_json_path, output_csv_path):
    """
    主函数
    
    参数:
        traj_csv_path: str, 轨迹CSV路径
        graph_json_path: str, graph.json文件路径
        output_csv_path: str, 输出CSV路径
    """
    print("🚀 开始统计车道段交通状况...")
    
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
    
    # 过滤掉没有车道段ID的记录
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna()].copy()
    traj_df = traj_df[traj_df['FID'].astype(str).str.strip() != ''].copy()
    filtered_count = len(traj_df)
    print(f"📊 过滤后保留 {filtered_count} 条有效记录（过滤前: {original_count}）")
    
    # 确保FID为字符串类型
    traj_df['FID'] = traj_df['FID'].astype(str).str.strip()
    
    # 加载图结构
    graph_dict = load_graph(graph_json_path)
    
    # =================== Step 2: 确定时间窗口 ===================
    print("🕐 正在确定时间窗口...")
    min_frame = traj_df['frame'].min()
    max_frame = traj_df['frame'].max()
    
    # 生成时间窗口（每1秒一个窗口）
    time_windows = []
    current_start = min_frame
    while current_start <= max_frame:
        time_windows.append((current_start, current_start + TIME_WINDOW))
        current_start += TIME_WINDOW
    
    print(f"✅ 共生成 {len(time_windows)} 个时间窗口（{min_frame:.2f} ~ {max_frame:.2f}）")
    
    # =================== Step 3: 按车道段和时间窗口统计 ===================
    print("📊 正在统计每个车道段在每个时间窗口的交通状况...")
    
    results = []
    
    # 获取所有车道段ID
    all_lane_ids = set(graph_dict.keys())
    
    # 将FID转换为整数，方便匹配
    traj_df['FID_int'] = traj_df['FID'].apply(lambda x: int(float(x)) if x else -1)
    
    # 对每个车道段和每个时间窗口进行统计
    for lane_id in all_lane_ids:
        lane_group = traj_df[traj_df['FID_int'] == lane_id]
        
        # 对该车道段的每个时间窗口进行统计
        for window_start, window_end in time_windows:
            # 筛选该时间窗口内的数据
            window_data = lane_group[
                (lane_group['frame'] >= window_start) & 
                (lane_group['frame'] < window_end)
            ].copy()
            
            # 如果没有车辆经过，写入默认值
            if window_data.empty:
                results.append({
                    'lane_id': lane_id,
                    'start_frame': window_start,
                    'avg_speed': -1,
                    'avg_occupancy': 0,
                    'total_vehicles': 0,
                    'car_ratio': 0,
                    'medium_ratio': 0,
                    'heavy_ratio': 0,
                    'motorcycle_ratio': 0
                })
                continue
            
            # 统计基本信息
            unique_vehicles = window_data['id'].nunique()
            
            # 计算平均速度
            avg_speed = window_data['v'].mean()
            
            # 计算平均占用率（需要统计每一帧的占用率，然后求平均）
            frame_occupancies = []
            for frame, frame_group in window_data.groupby('frame'):
                occupancy = calculate_occupancy_rate(frame_group, LANE_LENGTH)
                frame_occupancies.append(occupancy)
            
            avg_occupancy = np.mean(frame_occupancies) if frame_occupancies else 0.0
            
            # 统计车辆类型数量和比例（按唯一车辆统计）
            car_type_counts = defaultdict(int)
            if 'car_type' in window_data.columns:
                # 对每个唯一车辆，获取其车辆类型（如果有多个记录，取第一个）
                for vehicle_id in window_data['id'].unique():
                    vehicle_data = window_data[window_data['id'] == vehicle_id]
                    if not vehicle_data.empty:
                        car_type = vehicle_data.iloc[0]['car_type']
                        if pd.notna(car_type):
                            car_type_str = str(car_type).lower().strip()
                            car_type_counts[car_type_str] += 1
            
            car_ratio = car_type_counts.get('car', 0) / unique_vehicles if unique_vehicles > 0 else 0.0
            medium_ratio = car_type_counts.get('medium', 0) / unique_vehicles if unique_vehicles > 0 else 0.0
            heavy_ratio = car_type_counts.get('heavy', 0) / unique_vehicles if unique_vehicles > 0 else 0.0
            motorcycle_ratio = car_type_counts.get('motorcycle', 0) / unique_vehicles if unique_vehicles > 0 else 0.0
            
            # 保存结果
            results.append({
                'lane_id': lane_id,
                'start_frame': window_start,
                'avg_speed': round(avg_speed, 2),
                'avg_occupancy': round(avg_occupancy, 2),
                'total_vehicles': unique_vehicles,
                'car_ratio': round(car_ratio, 2),
                'medium_ratio': round(medium_ratio, 2),
                'heavy_ratio': round(heavy_ratio, 2),
                'motorcycle_ratio': round(motorcycle_ratio, 2)
            })
    
    # =================== Step 4: 保存结果 ===================
    print(f"💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['lane_id', 'start_frame']).reset_index(drop=True)
    
    # =================== 归一化处理 ===================
    # avg_speed: -1保持为1（畅通无阻），其他按0~100归一化到0~1
    results_df['avg_speed'] = results_df['avg_speed'].apply(
        lambda x: 1.0 if x == -1 else round(min(max(x / 100.0, 0.0), 1.0), 2)
    )
    
    # total_vehicles: 按对数变换 + 归一化
    results_df['total_vehicles'] = results_df['total_vehicles'].apply(
        lambda x: round(np.log(1 + x) / np.log(15) , 2)
    )
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 统计结果已保存至: {output_csv_path}")
    print(f"📊 总计统计记录数: {len(results_df)}")
    print(f"📊 涉及车道段数: {results_df['lane_id'].nunique()}")
    print(f"📊 时间窗口数: {results_df['start_frame'].nunique()}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    TRAJ_CSV_PATH = r"../data/trajectory_with_laneid/d210291000.csv"  # 轨迹数据
    GRAPH_JSON_PATH = r"../data/road_graph/graph_40m.json"  # 图结构（如果不存在，尝试使用graph.json）
    OUTPUT_CSV = r"../data/lane_node_stats/d210291000_lane_node_stats.csv"  # 输出路径
    
    # 如果指定的graph.json不存在，尝试使用默认的graph.json
    if not os.path.exists(GRAPH_JSON_PATH):
        raise FileNotFoundError(f"❌ 图文件不存在: {GRAPH_JSON_PATH}")
    
    # 执行统计
    main(TRAJ_CSV_PATH, GRAPH_JSON_PATH, OUTPUT_CSV)

