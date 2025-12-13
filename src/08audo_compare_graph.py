# -*- coding: utf-8 -*-
"""
11compare_json.py

比较两个JSON文件，检查节点中的node_connections是否一致
节点顺序和数组顺序都不重要

输入：两个JSON文件路径
输出：打印有差异的节点的lane_id

调用方法：
python 08audo_compare_graph.py ../data/road_graph/d210240900_graph.json ../data/road_graph/d210240930_graph.json
"""

import json
import sys
from typing import Dict, Any, Set, List


def normalize_connections(connections: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """
    标准化node_connections，对数组进行排序以便比较
    
    参数:
        connections: node_connections字典
    
    返回:
        标准化后的字典（数组已排序）
    """
    normalized = {}
    for conn_type, conn_list in connections.items():
        normalized[conn_type] = sorted(conn_list)
    return normalized


def compare_nodes(file1_path: str, file2_path: str):
    """
    比较两个JSON文件中的节点
    
    参数:
        file1_path: 第一个JSON文件路径
        file2_path: 第二个JSON文件路径
    """
    # 读取两个JSON文件
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return
    
    # 将节点按lane_id组织成字典
    nodes1 = {node['lane_id']: node for node in data1.get('nodes', [])}
    nodes2 = {node['lane_id']: node for node in data2.get('nodes', [])}
    
    # 获取所有lane_id
    all_lane_ids = set(nodes1.keys()) | set(nodes2.keys())
    
    # 存储有差异的lane_id
    diff_lane_ids = []
    
    # 比较每个节点
    for lane_id in sorted(all_lane_ids):
        node1 = nodes1.get(lane_id)
        node2 = nodes2.get(lane_id)
        
        # 如果节点只在一个文件中存在
        if node1 is None:
            print(f"⚠️  lane_id {lane_id} 只在第二个文件中存在")
            diff_lane_ids.append(lane_id)
            continue
        if node2 is None:
            print(f"⚠️  lane_id {lane_id} 只在第一个文件中存在")
            diff_lane_ids.append(lane_id)
            continue
        
        # 获取node_connections
        conn1 = node1.get('node_connections', {})
        conn2 = node2.get('node_connections', {})
        
        # 标准化连接（排序数组）
        normalized_conn1 = normalize_connections(conn1)
        normalized_conn2 = normalize_connections(conn2)
        
        # 比较连接是否一致
        if normalized_conn1 != normalized_conn2:
            diff_lane_ids.append(lane_id)
    
    # 输出结果
    if diff_lane_ids:
        print(f"\n📊 发现 {len(diff_lane_ids)} 个节点的node_connections存在差异:")
        print("差异节点的lane_id:")
        for lane_id in sorted(diff_lane_ids):
            print(f"  - {lane_id}")
    else:
        print("\n✅ 所有节点的node_connections完全一致！")


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python 11compare_json.py <file1.json> <file2.json>")
        print("示例: python 11compare_json.py graph1.json graph2.json")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    print(f"📁 比较文件:")
    print(f"  文件1: {file1_path}")
    print(f"  文件2: {file2_path}")
    print()
    
    compare_nodes(file1_path, file2_path)


if __name__ == '__main__':
    main()

