# -*- coding: utf-8 -*-
"""
07build_graph.py

基于车道段变动统计CSV和lanes信息构建道路图结构

输入：
    - INPUT_JSON: 包含lanes信息的JSON文件
    - TRANSITIONS_CSV_PATH: CSV文件，包含 from_lane_id, to_lane_id, count 三列
      （注意：这里的from_lane_id和to_lane_id实际上是node_id）
输出：JSON格式的图结构，包含原始lanes部分和新生成的nodes部分

规则：
- 保留原始lanes信息不变
- 从transitions CSV中获取node_id
- 根据lanes中的node列表找到每个node_id对应的lane_id
- count 最高的 to_lane_id 放到 direct 里
- 低于最高但仍大于0的放进 near 里
- nodes按node_id从小到大排序
"""

import os
import json
import pandas as pd

# =================== 可调整参数 ===================
SEGMENT_LENGTH = 10.0  # 段长度，开发人员可根据需要调整


def build_node_to_lane_mapping(lanes_data):
    """
    根据lanes信息构建node_id到lane_id的映射
    
    参数:
        lanes_data: list, lanes列表，每个元素包含lane_id和nodes
    返回:
        dict, node_id -> lane_id 的映射
    """
    node_to_lane = {}
    for lane in lanes_data:
        lane_id = lane['lane_id']
        for node_id in lane['nodes']:
            node_to_lane[node_id] = lane_id
    return node_to_lane


def main(input_json_path, transitions_csv_path, output_json_path):
    """
    主函数

    参数:
        input_json_path: str, 包含lanes信息的JSON文件路径
        transitions_csv_path: str, 车道段变动统计CSV路径，需包含 from_lane_id, to_lane_id, count 三列
        output_json_path: str, 输出 JSON 文件路径
    """
    print("🚀 开始基于车道段变动统计构建图结构...")

    # =================== Step 1: 读取lanes信息 ===================
    print("📦 正在读取lanes信息...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        lanes_json = json.load(f)
    
    lanes_data = lanes_json.get('lanes', [])
    print(f"✅ 共读取 {len(lanes_data)} 条lane记录")
    
    # 构建node_id到lane_id的映射
    node_to_lane = build_node_to_lane_mapping(lanes_data)
    print(f"✅ 构建了 {len(node_to_lane)} 个node到lane的映射")

    # =================== Step 2: 读取变动统计数据 ===================
    print("📦 正在读取车道段变动统计数据...")
    transitions_df = pd.read_csv(transitions_csv_path)
    
    # 检查必要字段
    required_fields = ['from_lane_id', 'to_lane_id', 'count']
    missing_fields = [f for f in required_fields if f not in transitions_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 变动统计数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(transitions_df)} 条变动记录")
    
    # 确保数据类型正确（CSV中的from_lane_id和to_lane_id实际上是node_id）
    transitions_df['from_node_id'] = transitions_df['from_lane_id'].astype(float).astype(int)
    transitions_df['to_node_id'] = transitions_df['to_lane_id'].astype(float).astype(int)
    transitions_df['count'] = transitions_df['count'].astype(int)
    
    # =================== Step 3: 按 from_node_id 分组处理 ===================
    print("🔍 正在处理每个节点的连接关系...")
    
    nodes_dict = {}  # 用字典存储，方便按node_id排序
    noise_count = 0  # 统计被舍弃的噪声数量
    
    # 按 from_node_id 分组
    for from_node_id, group in transitions_df.groupby('from_node_id'):
        # 按 count 降序排序
        sorted_group = group.sort_values('count', ascending=False)
        
        # 仅保留 count > 0 的有效记录
        valid_group = sorted_group[sorted_group['count'] > 0]
        noise_count += len(sorted_group) - len(valid_group)
        if valid_group.empty:
            continue
        
        # 获取最高 count 值
        max_count = valid_group.iloc[0]['count']
        
        # 初始化连接列表
        direct_connections = []
        near_connections = []
        
        # 遍历所有有效 to_node_id
        for _, row in valid_group.iterrows():
            to_node_id = row['to_node_id']
            count = row['count']
            
            # count 最高的放入 direct（确保是整数）
            if count == max_count:
                direct_connections.append(int(to_node_id))
            else:
                near_connections.append(int(to_node_id))
        
        # 构建节点连接字典
        node_connections = {}
        if direct_connections:
            node_connections["direct"] = direct_connections
        if near_connections:
            node_connections["near"] = near_connections
        
        # 获取lane_id（从映射中查找）
        lane_id = node_to_lane.get(from_node_id, None)
        if lane_id is None:
            print(f"⚠️ 警告: node_id {from_node_id} 未在lanes中找到对应的lane_id")
        
        # 添加到节点字典中
        nodes_dict[from_node_id] = {
            "node_id": from_node_id,
            "lane_id": lane_id,
            "position_in_lane": None,  # 先空着不填
            "segment_length": SEGMENT_LENGTH,
            "node_connections": node_connections
        }
    
    # 按node_id从小到大排序
    sorted_node_ids = sorted(nodes_dict.keys())
    nodes_list = [nodes_dict[node_id] for node_id in sorted_node_ids]
    
    print(f"✅ 共构建 {len(nodes_list)} 个节点")
    
    # 统计连接信息
    total_direct = sum(len(node.get('node_connections', {}).get('direct', [])) for node in nodes_list)
    total_near = sum(len(node.get('node_connections', {}).get('near', [])) for node in nodes_list)
    print(f"📊 direct 连接数: {total_direct}, near 连接数: {total_near}, 噪声舍弃数: {noise_count}")
    
    # =================== Step 4: 构建输出结构 ===================
    # 保留原始lanes部分，添加nodes部分
    output_data = {
        "lanes": lanes_data,
        "nodes": nodes_list
    }
    
    # =================== Step 5: 输出图结构 ===================
    print(f"💾 正在保存图结构到 {output_json_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # 保存JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 图结构已保存至: {output_json_path}")
    print(f"📊 总计lanes数: {len(lanes_data)}, 总计nodes数: {len(nodes_list)}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    INPUT_JSON = r"../data/road_graph/d210291000_lanes.json"  # 包含lanes信息的JSON
    TRANSITIONS_CSV_PATH = r"../data/road_graph/d210291000_transitions.csv"  # 车道段变动统计CSV
    OUTPUT_JSON = r"../data/road_graph/d210291000_graph.json"  # 输出路径

    # 执行构建
    main(INPUT_JSON, TRANSITIONS_CSV_PATH, OUTPUT_JSON)
