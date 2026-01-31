# 根据方向掩码信息和路网结构信息生成节点的mask文件
import json
import pandas as pd
import os

def load_graph(graph_path):
    """加载路网图数据"""
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    return graph

def build_direction_to_nodes_mapping(graph):
    """
    构建从方向ID到节点ID列表的映射
    
    返回: dict {direction_id: [node_ids]}
    """
    direction_to_nodes = {}
    
    # 遍历所有方向
    for direction in graph['directions']:
        direction_id = direction['direction_id']
        lane_ids = direction['lanes']  # 该方向包含的车道ID列表
        
        # 收集该方向所有车道的所有节点
        nodes_set = set()
        for lane_id in lane_ids:
            # 在lanes列表中找到对应的lane
            for lane in graph['lanes']:
                if lane['lane_id'] == lane_id:
                    # 获取该车道的所有节点
                    if 'nodes' in lane:
                        nodes_set.update(lane['nodes'])
                    break
        
        direction_to_nodes[direction_id] = sorted(list(nodes_set))
    
    return direction_to_nodes

def merge_time_intervals(node_visibility):
    """
    将节点的可见时间段合并成连续区间
    
    参数:
        node_visibility: dict {node_id: [frames]} 每个节点可见的帧列表
    
    返回:
        list of dict: [{node_id, start, end}, ...]
    """
    result = []
    
    for node_id in sorted(node_visibility.keys()):
        frames = sorted(node_visibility[node_id])
        
        if not frames:
            continue
        
        # 合并连续的帧为区间
        start = frames[0]
        end = frames[0]
        
        for i in range(1, len(frames)):
            if frames[i] == end + 1:
                # 连续，扩展当前区间
                end = frames[i]
            else:
                # 不连续，保存当前区间，开始新区间
                result.append({
                    'node_id': node_id,
                    'start': start,
                    'end': end
                })
                start = frames[i]
                end = frames[i]
        
        # 保存最后一个区间
        result.append({
            'node_id': node_id,
            'start': start,
            'end': end
        })
    
    return result

def generate_node_mask(patrol_mask_path, graph_path, output_path):
    """
    生成节点级别的mask文件（格式：node_id, start, end）
    
    参数:
        patrol_mask_path: patrol_mask_relative.csv 文件路径
        graph_path: graph_10m.json 文件路径
        output_path: 输出文件路径
    """
    # 1. 加载数据
    print("加载数据...")
    patrol_mask = pd.read_csv(patrol_mask_path)
    graph = load_graph(graph_path)
    
    # 2. 构建方向到节点的映射
    print("构建方向到节点的映射...")
    direction_to_nodes = build_direction_to_nodes_mapping(graph)
    
    # 打印映射信息
    print("\n方向到节点的映射:")
    for direction_id, nodes in direction_to_nodes.items():
        print(f"  {direction_id}: {len(nodes)} 个节点 - {nodes}")
    
    # 3. 获取所有节点ID
    all_nodes = set()
    for nodes in direction_to_nodes.values():
        all_nodes.update(nodes)
    all_nodes = sorted(list(all_nodes))
    print(f"\n总共 {len(all_nodes)} 个节点")
    
    # 4. 确定时间范围
    max_frame = patrol_mask['end'].max()
    print(f"最大帧数: {max_frame}")
    
    # 5. 根据patrol_mask直接映射生成节点可见时间段
    print("\n生成节点可见时间段...")
    node_visibility = {node_id: [] for node_id in all_nodes}
    
    # 遍历每个方向的时间段
    for _, row in patrol_mask.iterrows():
        direction_id = row['direction_id']
        start = row['start']
        end = row['end']
        
        # 获取该方向包含的所有节点
        if direction_id in direction_to_nodes:
            nodes = direction_to_nodes[direction_id]
            # 为这些节点添加可见帧
            for node_id in nodes:
                for frame in range(start, end + 1):
                    if frame not in node_visibility[node_id]:
                        node_visibility[node_id].append(frame)
    
    # 6. 合并连续的时间段
    print("合并连续时间段...")
    mask_data = merge_time_intervals(node_visibility)
    
    # 7. 保存结果
    print(f"\n保存结果到 {output_path}...")
    mask_df = pd.DataFrame(mask_data)
    mask_df.to_csv(output_path, index=False)
    
    print(f"完成！生成了 {len(mask_df)} 条记录")
    print(f"节点数量: {len(all_nodes)}")
    print(f"时间范围: 0 到 {max_frame}")

if __name__ == '__main__':
    # 设置文件路径
    patrol_mask_path = '../data/lane_node_stats/patrol_mask_relative.csv'
    graph_path = '../data/road_graph/graph_10m.json'
    output_path = '../data/lane_node_stats/d210291000_node_mask.csv'
    
    # 生成节点mask
    generate_node_mask(patrol_mask_path, graph_path, output_path)
