#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从简化的路网定义自动构建完整的graph格式
消除冗余，只需要最小信息量
"""

import json
import sys
from typing import Dict, List, Any
from collections import defaultdict


def build_full_graph(simplified_graph: Dict) -> Dict:
    """
    从简化格式构建完整的graph格式
    """
    # 复制基础信息
    full_graph = {
        'intersection': simplified_graph['intersection'],
        'signal_timing': simplified_graph['signal_timing'],
        'directions': simplified_graph['directions']
    }
    
    # 构建车道ID到节点列表的映射
    lane_to_nodes = {}
    for lane in simplified_graph['lanes']:
        # 根据min_node、max_node、order生成节点列表
        min_node = lane['min_node']
        max_node = lane['max_node']
        order = lane.get('order', 'asc')  # 默认升序
        
        if order == 'asc':
            nodes = list(range(min_node, max_node + 1))
        else:  # desc
            nodes = list(range(max_node, min_node - 1, -1))
        
        lane_to_nodes[lane['lane_id']] = nodes
    
    # 构建车道连接关系（自动反向生成upstream_sources）
    lane_connections = {}  # lane_id -> {upstream: [], downstream: []}
    for lane in simplified_graph['lanes']:
        lane_id = lane['lane_id']
        lane_connections[lane_id] = {'upstream': [], 'downstream': []}
        
        if 'connections' in lane:
            for conn in lane['connections']:
                lane_connections[lane_id]['downstream'].append(conn)
                # 反向添加到目标车道的upstream
                target_lane = conn['to_lane']
                if target_lane not in lane_connections:
                    lane_connections[target_lane] = {'upstream': [], 'downstream': []}
                lane_connections[target_lane]['upstream'].append({
                    'from_lane': lane_id,
                    'movement': conn['movement'],
                    'travel_time': conn['travel_time'],
                    'ratio': conn['ratio']
                })
    
    # 构建完整的lanes数组
    full_lanes = []
    for lane in simplified_graph['lanes']:
        lane_id = lane['lane_id']
        nodes = lane_to_nodes[lane_id]
        
        # 构建完整车道信息
        full_lane = {
            'lane_id': lane_id,
            'movements': lane['movements'],
            'nodes': nodes,  # 使用生成的节点列表
            'stopline_node': lane.get('stopline_node'),
            'total_length': lane['total_length'],
            'segment_length': lane['segment_length']
        }
        
        # 添加lane_type和连接信息
        has_downstream = len(lane_connections[lane_id]['downstream']) > 0
        has_upstream = len(lane_connections[lane_id]['upstream']) > 0
        
        if has_downstream and not has_upstream:
            full_lane['lane_type'] = 'upstream'
            full_lane['downstream_connections'] = [
                {
                    'target_lane': str(conn['to_lane']),
                    'movement': conn['movement'],
                    'travel_time': conn['travel_time'],
                    'split_ratio': conn['ratio']
                }
                for conn in lane_connections[lane_id]['downstream']
            ]
        elif has_upstream and not has_downstream:
            full_lane['lane_type'] = 'downstream'
            full_lane['upstream_sources'] = [
                {
                    'source_lane': str(conn['from_lane']),
                    'movement': conn['movement'],
                    'travel_time': conn['travel_time'],
                    'contribution_ratio': conn['ratio']
                }
                for conn in lane_connections[lane_id]['upstream']
            ]
        elif has_upstream and has_downstream:
            # 中间车道，既有上游也有下游
            full_lane['lane_type'] = 'intermediate'
            full_lane['downstream_connections'] = [
                {
                    'target_lane': str(conn['to_lane']),
                    'movement': conn['movement'],
                    'travel_time': conn['travel_time'],
                    'split_ratio': conn['ratio']
                }
                for conn in lane_connections[lane_id]['downstream']
            ]
            full_lane['upstream_sources'] = [
                {
                    'source_lane': str(conn['from_lane']),
                    'movement': conn['movement'],
                    'travel_time': conn['travel_time'],
                    'contribution_ratio': conn['ratio']
                }
                for conn in lane_connections[lane_id]['upstream']
            ]
        
        full_lanes.append(full_lane)
    
    full_graph['lanes'] = full_lanes
    
    # 构建完整的nodes数组（自动生成节点连接）
    full_nodes = build_full_nodes(
        simplified_graph['lanes'],
        lane_to_nodes,
        simplified_graph.get('adjacent_lanes', [])
    )
    
    full_graph['nodes'] = full_nodes
    
    # 构建phase_conflicts
    phase_conflicts = {}
    for direction in simplified_graph['directions']:
        phase_id = direction['signal_phase']
        if phase_id not in phase_conflicts:
            phase_conflicts[phase_id] = direction.get('conflict_phases', [])
    
    full_graph['phase_conflicts'] = phase_conflicts
    
    return full_graph


def build_full_nodes(lanes_data: List[Dict], lane_to_nodes: Dict[int, List[int]], 
                     adjacent_lanes: List[List[int]]) -> List[Dict]:
    """
    构建完整的节点数组，包括节点连接关系
    
    Args:
        lanes_data: 车道数据列表
        lane_to_nodes: 车道ID到节点列表的映射 {lane_id: [node_ids]}
        adjacent_lanes: 相邻车道分组 [[0,1], [2,3,4]]
    """
    # 创建车道邻接关系映射
    lane_neighbors = {}
    for group in adjacent_lanes:
        for lane_id in group:
            lane_neighbors[lane_id] = [l for l in group if l != lane_id]
    
    # 创建节点到车道的反向映射
    node_to_lane = {}  # node_id -> (lane_id, position_idx)
    for lane_id, node_list in lane_to_nodes.items():
        for pos_idx, node_id in enumerate(node_list):
            node_to_lane[node_id] = (lane_id, pos_idx)
    
    # 创建车道信息映射
    lane_info = {}
    for lane_id, node_list in lane_to_nodes.items():
        # 从lanes_data中找到对应的车道
        lane_data = next((l for l in lanes_data if l['lane_id'] == lane_id), None)
        if lane_data:
            lane_info[lane_id] = {
                'segment_length': lane_data['segment_length'],
                'total_length': lane_data['total_length'],
                'nodes': node_list,
                'stopline_node': lane_data.get('stopline_node')
            }
    
    # 构建所有节点
    full_nodes = []
    all_node_ids = sorted(node_to_lane.keys())
    
    for node_id in all_node_ids:
        lane_id, pos_idx = node_to_lane[node_id]
        lane_nodes = lane_info[lane_id]['nodes']
        segment_length = lane_info[lane_id]['segment_length']
        stopline_node = lane_info[lane_id].get('stopline_node')
        
        # 计算position_in_lane
        position_in_lane = pos_idx * segment_length
        
        # 如果是stopline_node，position_in_lane设为None
        if stopline_node is not None and node_id == stopline_node:
            position_in_lane = None
        
        # 构建节点连接
        node_connections = {'direct': [], 'near': []}
        
        # Direct连接：同车道的下一个节点
        if pos_idx < len(lane_nodes) - 1:
            next_node_id = lane_nodes[pos_idx + 1]
            node_connections['direct'].append(next_node_id)
        
        # Near连接：相邻车道的相同位置节点
        if lane_id in lane_neighbors:
            for neighbor_lane_id in lane_neighbors[lane_id]:
                if neighbor_lane_id in lane_to_nodes:
                    neighbor_nodes = lane_to_nodes[neighbor_lane_id]
                    # 如果相邻车道有相同位置的节点，建立near连接
                    if pos_idx < len(neighbor_nodes):
                        neighbor_node_id = neighbor_nodes[pos_idx]
                        node_connections['near'].append(neighbor_node_id)
        
        # 构建节点
        node = {
            'node_id': node_id,
            'lane_id': lane_id,
            'position_in_lane': position_in_lane,
            'segment_length': segment_length,
            'node_connections': node_connections
        }
        
        full_nodes.append(node)
    
    return full_nodes


def main():
    if len(sys.argv) < 2:
        print("用法: python build_full_graph.py <simplified_graph.json> [output.json]")
        print("示例: python build_full_graph.py data/road_graph/simplified_graph.json data/road_graph/full_graph.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('simplified', 'full')
    
    # 读取简化格式
    with open(input_file, 'r', encoding='utf-8') as f:
        simplified_graph = json.load(f)
    
    # 构建完整格式
    full_graph = build_full_graph(simplified_graph)
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_graph, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 成功生成完整路网文件: {output_file}")
    print(f"  - 车道数: {len(full_graph['lanes'])}")
    print(f"  - 节点数: {len(full_graph['nodes'])}")
    print(f"  - 方向数: {len(full_graph['directions'])}")


if __name__ == '__main__':
    main()
