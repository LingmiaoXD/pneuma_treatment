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

def generate_node_mask(patrol_mask_path, graph_path, output_path):
    """
    生成节点级别的mask文件
    
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
    max_frame = patrol_mask['relative_end'].max()
    print(f"最大帧数: {max_frame}")
    
    # 5. 生成mask数据
    print("\n生成mask数据...")
    mask_data = []
    
    # 对每一帧
    for frame in range(0, max_frame + 1):
        # 找出当前帧哪些方向是可见的
        visible_directions = patrol_mask[
            (patrol_mask['relative_start'] <= frame) & 
            (patrol_mask['relative_end'] >= frame)
        ]['direction_id'].tolist()
        
        # 收集可见方向的所有节点
        visible_nodes = set()
        for direction_id in visible_directions:
            if direction_id in direction_to_nodes:
                visible_nodes.update(direction_to_nodes[direction_id])
        
        # 为每个节点生成记录
        for node_id in all_nodes:
            is_observed = 1 if node_id in visible_nodes else 0
            mask_data.append({
                'start_frame': frame,
                'node_id': node_id,
                'is_observed': is_observed
            })
    
    # 6. 保存结果
    print(f"\n保存结果到 {output_path}...")
    mask_df = pd.DataFrame(mask_data)
    mask_df.to_csv(output_path, index=False)
    
    print(f"完成！生成了 {len(mask_df)} 条记录")
    print(f"帧数范围: 0 到 {max_frame}")
    print(f"节点数量: {len(all_nodes)}")

if __name__ == '__main__':
    # 设置文件路径
    patrol_mask_path = '../data/lane_node_stats/patrol_mask_relative.csv'
    graph_path = '../data/road_graph/graph_10m.json'
    output_path = '../data/lane_node_stats/d210291000_node_mask.csv'
    
    # 生成节点mask
    generate_node_mask(patrol_mask_path, graph_path, output_path)
