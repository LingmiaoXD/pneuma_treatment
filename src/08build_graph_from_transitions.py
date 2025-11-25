# -*- coding: utf-8 -*-
"""
08build_graph_from_transitions.py

基于车道段变动统计CSV构建道路图结构

输入：CSV文件，包含 from_lane_id, to_lane_id, count 三列
输出：JSON格式的图结构，与06make_node.py格式一致

规则：
- 每个 from_lane_id 作为一个 node 的 lane_id
- count 最高的 to_lane_id 放到 direct 里
- 低于最高但通过IQR方法筛选的放进 near 里（使用Q1作为阈值，低于Q1的作为噪声舍弃）
- 其余的作为噪声舍弃
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict


def main(transitions_csv_path, output_json_path):
    """
    主函数

    参数:
        transitions_csv_path: str, 车道段变动统计CSV路径，需包含 from_lane_id, to_lane_id, count 三列
        output_json_path: str, 输出 JSON 文件路径
    """
    print("🚀 开始基于车道段变动统计构建图结构...")

    # =================== Step 1: 读取变动统计数据 ===================
    print("📦 正在读取车道段变动统计数据...")
    transitions_df = pd.read_csv(transitions_csv_path)
    
    # 检查必要字段
    required_fields = ['from_lane_id', 'to_lane_id', 'count']
    missing_fields = [f for f in required_fields if f not in transitions_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 变动统计数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(transitions_df)} 条变动记录")
    
    # 确保数据类型正确
    transitions_df['from_lane_id'] = transitions_df['from_lane_id'].astype(str).str.strip()
    transitions_df['to_lane_id'] = transitions_df['to_lane_id'].astype(str).str.strip()
    transitions_df['count'] = transitions_df['count'].astype(int)
    
    # =================== Step 2: 按 from_lane_id 分组处理 ===================
    print("🔍 正在处理每个车道段的连接关系...")
    
    graph_data = {"nodes": []}
    noise_count = 0  # 统计被舍弃的噪声数量
    
    # 按 from_lane_id 分组
    for from_lane_id, group in transitions_df.groupby('from_lane_id'):
        # 按 count 降序排序
        sorted_group = group.sort_values('count', ascending=False)
        
        # 获取最高 count 值
        max_count = sorted_group.iloc[0]['count']
        
        # 使用IQR方法计算阈值（针对该from_lane_id的所有count值）
        counts = sorted_group['count'].values
        
        # 如果只有1个或2个数据点，IQR不适用，使用简单规则
        if len(counts) <= 2:
            # 如果只有1个，直接作为direct；如果有2个，最高的作为direct，另一个作为near
            iqr_threshold = 0
        else:
            # 计算Q1、Q3和IQR
            Q1 = np.percentile(counts, 25)
            Q3 = np.percentile(counts, 75)
            IQR = Q3 - Q1
            
            # 使用Q1作为阈值（低于Q1的作为噪声）
            # 对于"少数高频+多数低频"的分布，Q1以下通常是噪声
            iqr_threshold = Q1
        
        # 初始化连接列表
        direct_connections = []
        near_connections = []
        
        # 遍历所有可能的 to_lane_id
        for _, row in sorted_group.iterrows():
            to_lane_id = row['to_lane_id']
            count = row['count']
            
            # count 最高的放入 direct
            if count == max_count:
                direct_connections.append(int(to_lane_id))
            # 低于最高但大于等于IQR阈值的放入 near
            elif count >= iqr_threshold:
                near_connections.append(int(to_lane_id))
            # 其余舍弃（作为噪声）
            else:
                noise_count += 1
        
        # 构建节点连接字典
        node_connections = {}
        if direct_connections:
            node_connections["direct"] = direct_connections
        if near_connections:
            node_connections["near"] = near_connections
        
        # 添加到图结构中
        graph_data["nodes"].append({
            "lane_id": str(from_lane_id),
            "node_connections": node_connections
        })
    
    print(f"✅ 共构建 {len(graph_data['nodes'])} 个节点")
    
    # 统计连接信息
    total_direct = sum(len(node.get('node_connections', {}).get('direct', [])) for node in graph_data['nodes'])
    total_near = sum(len(node.get('node_connections', {}).get('near', [])) for node in graph_data['nodes'])
    print(f"📊 direct 连接数: {total_direct}, near 连接数: {total_near}, 噪声舍弃数: {noise_count}")
    
    # =================== Step 3: 输出图结构 ===================
    print(f"💾 正在保存图结构到 {output_json_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # 保存JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 图结构已保存至: {output_json_path}")
    print(f"📊 总计节点数: {len(graph_data['nodes'])}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    TRANSITIONS_CSV_PATH = r"../data/road_graph/d210240830_transitions.csv"  # 车道段变动统计CSV
    OUTPUT_JSON = r"../data/road_graph/d210240830_graph.json"  # 输出路径

    # 执行构建
    main(TRANSITIONS_CSV_PATH, OUTPUT_JSON)

