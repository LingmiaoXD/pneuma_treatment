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
- 轨迹CSV，包含 id, corrected_x, corrected_y, width, height, speed_kmh, start_time, FID, lane_id, car_type 等字段
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

# 车辆占用长度现在使用width字段（车辆最长边），不再使用固定的车辆类型长度

# 滑动时间窗口大小（秒）- 为不同指标设置不同的窗口
SPEED_WINDOW = 2.0       # 速度滑块：2秒（捕捉瞬时速度变化）
FLOW_WINDOW = 2.0       # 流量滑块：2秒（累积足够的车辆数）
OCCUPANCY_WINDOW = 2.0   # 占用率滑块：2秒（平衡敏感度和稳定性）

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


def get_vehicle_length(width):
    """
    获取车辆占用长度（使用width字段）
    
    参数:
        width: float, 车辆最长边（米）
        
    返回:
        float: 占用长度（米）
    """
    if pd.isna(width) or width is None or width <= 0:
        # 如果width未知或无效，使用默认值4.0米
        return 4.0
    
    return float(width)


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
    for _, row in future_traj.iterrows():
        next_node_id = row['FID']
        if pd.notna(next_node_id) and int(next_node_id) != current_node_id:
            return int(next_node_id)
    
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


def calculate_occupancy_rate(group, segment_length, current_node_id, node_dict, traj_df, current_frame):
    """
    计算占用率（考虑车辆长度对下一节点的影响）
    
    车辆占用分配策略：
    - 当前节点（车辆中心点所在节点）：占用车辆长度的 3/4
    - 下一个节点（direct连接的节点）：占用车辆长度的 1/4
    
    参数:
        group: DataFrame, 某一帧内当前节点的所有车辆记录
        segment_length: float, 节点段长度（米）
        current_node_id: int, 当前节点ID
        node_dict: dict, 节点字典
        traj_df: DataFrame, 完整轨迹数据
        current_frame: float, 当前时间帧
        
    返回:
        float: 占用率（0-1之间）
    """
    if group.empty:
        return 0.0
    
    # 计算当前节点的占用长度
    total_length = 0.0
    
    for _, row in group.iterrows():
        width = row.get('width')
        vehicle_length = get_vehicle_length(width)
        
        # 当前节点占用3/4的车辆长度
        total_length += vehicle_length * 0.75
        
        # 尝试找到下一个节点，判断是否需要将1/4分配给下一节点
        vehicle_id = row['id']
        next_node_id = get_next_node_for_vehicle(traj_df, vehicle_id, current_node_id, current_frame)
        
        # 如果找到了下一个节点，并且它在direct连接中，则将1/4分配给下一节点
        # （这部分占用会在计算下一节点时被加上）
        if next_node_id is not None and current_node_id in node_dict:
            node_info = node_dict[current_node_id]
            if next_node_id in node_info['direct']:
                # 这里只是记录，实际的1/4占用会在下一节点计算时加上
                pass
    
    # 查找所有在当前帧中，下一个节点是当前节点的车辆（即前一节点的车辆）
    # 这些车辆会贡献1/4的长度给当前节点
    frame_data = traj_df[traj_df['frame'] == current_frame]
    for _, row in frame_data.iterrows():
        vehicle_id = row['id']
        vehicle_node_id = int(row['FID']) if pd.notna(row['FID']) else None
        
        # 跳过当前节点的车辆（已经在上面计算过了）
        if vehicle_node_id == current_node_id:
            continue
        
        # 检查这辆车的下一个节点是否是当前节点
        next_node_id = get_next_node_for_vehicle(traj_df, vehicle_id, vehicle_node_id, current_frame)
        
        if next_node_id == current_node_id and vehicle_node_id in node_dict:
            # 检查是否是direct连接
            prev_node_info = node_dict[vehicle_node_id]
            if current_node_id in prev_node_info['direct']:
                # 这辆车在前一个节点，但会占用当前节点1/4的长度
                width = row.get('width')
                vehicle_length = get_vehicle_length(width)
                total_length += vehicle_length * 0.25
    
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
    
    # 检查必要字段（新的列名）
    required_fields = ['id', 'start_time', 'FID', 'speed_kmh', 'width']
    missing_fields = [f for f in required_fields if f not in traj_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 轨迹数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(traj_df)} 条轨迹记录")
    
    # 数据类型转换：处理可能是文本格式的数字字段
    print("🔄 正在转换数据类型...")
    numeric_fields = ['id', 'start_time', 'corrected_x', 'corrected_y', 'width', 'height', 'speed_kmh', 'FID', 'lane_id']
    for field in numeric_fields:
        if field in traj_df.columns:
            # 先转为字符串，去除可能的分号、空格、引号等
            traj_df[field] = traj_df[field].astype(str).str.strip().str.rstrip(';').str.strip('"').str.strip("'")
            # 替换空字符串为NaN
            traj_df[field] = traj_df[field].replace('', np.nan)
            traj_df[field] = traj_df[field].replace('nan', np.nan)
            # 转换为数值类型，无法转换的设为NaN
            traj_df[field] = pd.to_numeric(traj_df[field], errors='coerce')
    
    print(f"✅ 数据类型转换完成")
    
    # 将start_time重命名为frame，保持后续代码兼容
    traj_df['frame'] = traj_df['start_time']
    
    # 过滤掉没有节点ID的记录
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna()].copy()
    filtered_count = len(traj_df)
    print(f"📊 过滤后保留 {filtered_count} 条有效记录（过滤前: {original_count}）")
    
    # 加载图结构
    node_dict = load_graph(graph_json_path)
    
    # =================== Step 2: 确定时间窗口 ===================
    print("🕐 正在确定时间窗口...")
    min_frame = traj_df['frame'].min()
    max_frame = traj_df['frame'].max()
    
    # 使用最大窗口半径来确定输出范围，确保所有指标都有完整的滑动窗口数据
    # 向上取整到整数秒，确保输出时间点是整数
    output_start = int(np.ceil(min_frame + MAX_HALF_WINDOW))
    output_end = int(np.floor(max_frame - MAX_HALF_WINDOW))
    
    # 生成输出时间点（每1秒一个，都是整数）
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
    
    # 将FID转换为整数，方便匹配（FID已经是数值类型）
    traj_df['FID_int'] = traj_df['FID'].apply(lambda x: int(x) if pd.notna(x) else -1)
    
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
                avg_speed = speed_window_data['speed_kmh'].abs().mean()
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
                    occupancy = calculate_occupancy_rate(
                        frame_group, segment_length, node_id, node_dict, 
                        traj_df, frame
                    )
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
        lambda x: round(np.log(1 + x) / np.log(8) , 2)
    )
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 统计结果已保存至: {output_csv_path}")
    print(f"📊 总计统计记录数: {len(results_df)}")
    print(f"📊 涉及节点数: {results_df['node_id'].nunique()}")
    print(f"📊 时间窗口数: {results_df['start_frame'].nunique()}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    TRAJ_CSV_PATH = r"/home/nvme1/pneuma/data/trajectory_with_laneid/0129094705_0001.csv"  # 轨迹数据
    GRAPH_JSON_PATH = r"/home/nvme1/pneuma/data/road_graph/minhang_graph.json"  # 图结构（更新版本，包含lanes和nodes）
    OUTPUT_CSV = r"/home/nvme1/pneuma/data/lane_node_stats/k0129094705_0001_lane_node_stats_3.csv"  # 输出路径
    
    if not os.path.exists(GRAPH_JSON_PATH):
        raise FileNotFoundError(f"❌ 图文件不存在: {GRAPH_JSON_PATH}")
    
    # 执行统计
    main(TRAJ_CSV_PATH, GRAPH_JSON_PATH, OUTPUT_CSV)

