#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证路网文件的正确性
"""

import json
import sys
from collections import defaultdict


def validate_graph(graph_file):
    """验证路网文件"""
    
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    errors = []
    warnings = []
    
    print(f"\n{'='*60}")
    print(f"验证文件: {graph_file}")
    print(f"{'='*60}\n")
    
    # 1. 检查基本结构
    required_keys = ['intersection', 'signal_timing', 'directions', 'lanes', 'nodes']
    for key in required_keys:
        if key not in graph:
            errors.append(f"缺少必需字段: {key}")
    
    if errors:
        print("❌ 基本结构检查失败")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✓ 基本结构检查通过")
    
    # 2. 检查车道
    lanes = graph['lanes']
    lane_ids = set()
    lane_types = defaultdict(int)
    
    for lane in lanes:
        lane_id = lane['lane_id']
        
        # 检查重复ID
        if lane_id in lane_ids:
            errors.append(f"车道ID重复: {lane_id}")
        lane_ids.add(lane_id)
        
        # 统计车道类型
        lane_type = lane.get('lane_type', 'unknown')
        lane_types[lane_type] += 1
        
        # 检查节点数量
        expected_nodes = int(lane['total_length'] / lane['segment_length'])
        actual_nodes = len(lane['nodes'])
        if expected_nodes != actual_nodes:
            warnings.append(
                f"车道{lane_id}节点数不匹配: 期望{expected_nodes}, 实际{actual_nodes}"
            )
        
        # 检查连接
        if 'downstream_connections' in lane:
            for conn in lane['downstream_connections']:
                target = int(conn['target_lane'])
                if target not in lane_ids and target not in [l['lane_id'] for l in lanes]:
                    errors.append(f"车道{lane_id}连接到不存在的车道{target}")
    
    print(f"✓ 车道检查通过: {len(lanes)}条车道")
    print(f"  - upstream: {lane_types['upstream']}")
    print(f"  - downstream: {lane_types['downstream']}")
    print(f"  - intermediate: {lane_types.get('intermediate', 0)}")
    print(f"  - unknown: {lane_types.get('unknown', 0)}")
    
    # 3. 检查节点
    nodes = graph['nodes']
    node_ids = set()
    
    for node in nodes:
        node_id = node['node_id']
        
        # 检查重复ID
        if node_id in node_ids:
            errors.append(f"节点ID重复: {node_id}")
        node_ids.add(node_id)
        
        # 检查节点连接
        if 'node_connections' in node:
            for conn_type in ['direct', 'near']:
                if conn_type in node['node_connections']:
                    for target_node in node['node_connections'][conn_type]:
                        if target_node not in node_ids and target_node not in [n['node_id'] for n in nodes]:
                            warnings.append(
                                f"节点{node_id}的{conn_type}连接指向不存在的节点{target_node}"
                            )
    
    print(f"✓ 节点检查通过: {len(nodes)}个节点")
    
    # 4. 检查连接一致性
    print("\n检查连接一致性...")
    
    # 检查downstream -> upstream的一致性
    downstream_map = defaultdict(list)  # target_lane -> [(source_lane, movement, ratio)]
    
    for lane in lanes:
        lane_id = lane['lane_id']
        if 'downstream_connections' in lane:
            for conn in lane['downstream_connections']:
                target = int(conn['target_lane'])
                downstream_map[target].append({
                    'source': lane_id,
                    'movement': conn['movement'],
                    'ratio': conn['split_ratio']
                })
    
    # 验证upstream_sources
    for lane in lanes:
        lane_id = lane['lane_id']
        if 'upstream_sources' in lane:
            expected_sources = downstream_map.get(lane_id, [])
            actual_sources = lane['upstream_sources']
            
            if len(expected_sources) != len(actual_sources):
                warnings.append(
                    f"车道{lane_id}的upstream_sources数量不匹配: "
                    f"期望{len(expected_sources)}, 实际{len(actual_sources)}"
                )
            
            # 检查每个source是否匹配
            for expected in expected_sources:
                found = False
                for actual in actual_sources:
                    if (int(actual['source_lane']) == expected['source'] and
                        actual['movement'] == expected['movement']):
                        found = True
                        # 检查比例是否一致
                        if abs(actual['contribution_ratio'] - expected['ratio']) > 0.01:
                            warnings.append(
                                f"车道{lane_id}从车道{expected['source']}的比例不一致: "
                                f"downstream定义{expected['ratio']}, "
                                f"upstream定义{actual['contribution_ratio']}"
                            )
                        break
                
                if not found:
                    warnings.append(
                        f"车道{lane_id}缺少来自车道{expected['source']}的upstream_source"
                    )
    
    print("✓ 连接一致性检查完成")
    
    # 5. 检查分流比例
    print("\n检查分流比例...")
    
    for lane in lanes:
        lane_id = lane['lane_id']
        if 'downstream_connections' in lane:
            total_ratio = sum(conn['split_ratio'] for conn in lane['downstream_connections'])
            if abs(total_ratio - 1.0) > 0.01:
                warnings.append(
                    f"车道{lane_id}的分流比例总和不等于1.0: {total_ratio:.3f}"
                )
    
    print("✓ 分流比例检查完成")
    
    # 6. 输出结果
    print(f"\n{'='*60}")
    print("验证结果:")
    print(f"{'='*60}\n")
    
    if errors:
        print(f"❌ 发现 {len(errors)} 个错误:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("✓ 没有发现错误")
    
    if warnings:
        print(f"\n⚠️  发现 {len(warnings)} 个警告:")
        for warn in warnings:
            print(f"  - {warn}")
    else:
        print("\n✓ 没有发现警告")
    
    print(f"\n{'='*60}")
    
    # 7. 统计信息
    print("\n统计信息:")
    print(f"  - 路口: {graph['intersection']['name']}")
    print(f"  - 车道数: {len(lanes)}")
    print(f"  - 节点数: {len(nodes)}")
    print(f"  - 方向数: {len(graph['directions'])}")
    print(f"  - 信号相位数: {len(graph['signal_timing']['phases'])}")
    
    return len(errors) == 0


def main():
    if len(sys.argv) < 2:
        print("用法: python validate_graph.py <graph_file.json>")
        print("示例: python validate_graph.py data/road_graph/graph_10m.json")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    
    try:
        success = validate_graph(graph_file)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
