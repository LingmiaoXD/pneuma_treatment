# -*- coding: utf-8 -*-
"""
12nodemask2timemask.py

将节点级别的时间段mask转换为时间步级别的mask

输入：
    1. CSV文件，三列：node_id, start, end
       每一段内部都是连续的，start和end时刻左闭右开
    2. 图结构文件（用于获取所有的node_id）

输出：
    CSV文件，三列：time, node_id, is_observed
    每一行都是单个时间步、单个node_id
    输出的时候需要覆盖输入最小到最大所有的时刻、所有的node_id
"""

import os
import json
import pandas as pd


def load_all_node_ids(graph_json_path):
    """
    从图结构文件中加载所有的node_id
    
    参数:
        graph_json_path: str, graph.json文件路径
        
    返回:
        list: 所有node_id的列表
    """
    print(f"📦 正在读取图结构: {graph_json_path}")
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    node_ids = [n['node_id'] for n in graph_data.get('nodes', [])]
    print(f"✅ 共加载 {len(node_ids)} 个节点")
    
    return node_ids


def convert_node_mask_to_time_mask(input_csv_path, graph_json_path, output_csv_path):
    """
    将节点级别的时间段mask转换为时间步级别的mask
    
    参数:
        input_csv_path: str, 输入CSV路径（node_id, start, end）
        graph_json_path: str, 图结构文件路径
        output_csv_path: str, 输出CSV路径（time, node_id, is_observed）
    """
    print("🚀 开始转换节点mask到时间mask...")
    
    # =================== Step 1: 读取数据 ===================
    print("\n📦 正在读取输入数据...")
    mask_df = pd.read_csv(input_csv_path)
    
    # 检查必要字段
    required_fields = ['node_id', 'start', 'end']
    missing_fields = [f for f in required_fields if f not in mask_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 输入数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(mask_df)} 条时间段记录")
    
    # 加载所有node_id
    all_node_ids = load_all_node_ids(graph_json_path)
    
    # =================== Step 2: 确定时间范围 ===================
    min_time = int(mask_df['start'].min())
    max_time = int(mask_df['end'].max())
    
    print(f"\n📊 时间范围: {min_time} - {max_time}")
    print(f"📊 时间步数: {max_time - min_time}")
    print(f"📊 节点数: {len(all_node_ids)}")
    
    # =================== Step 3: 构建时间-节点的可见性映射 ===================
    print("\n🔄 正在构建可见性映射...")
    
    # 创建一个字典来存储每个(time, node_id)的可见性
    # 默认所有都是不可见(0)
    visibility_map = {}
    
    # 遍历所有时间步和节点，初始化为不可见
    for t in range(min_time, max_time):
        for node_id in all_node_ids:
            visibility_map[(t, node_id)] = 0
    
    # 根据输入的时间段，标记可见的时间步
    for _, row in mask_df.iterrows():
        node_id = row['node_id']
        start = int(row['start'])
        end = int(row['end'])
        
        # 左闭右开区间 [start, end)
        for t in range(start, end):
            if (t, node_id) in visibility_map:
                visibility_map[(t, node_id)] = 1
    
    print(f"✅ 可见性映射构建完成")
    
    # =================== Step 4: 生成输出数据 ===================
    print("\n📝 正在生成输出数据...")
    
    results = []
    for t in range(min_time, max_time):
        for node_id in all_node_ids:
            is_observed = visibility_map.get((t, node_id), 0)
            results.append({
                'time': t,
                'node_id': node_id,
                'is_observed': is_observed
            })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 统计信息
    total_records = len(results_df)
    observed_records = (results_df['is_observed'] == 1).sum()
    observed_ratio = observed_records / total_records * 100 if total_records > 0 else 0
    
    print(f"✅ 总记录数: {total_records}")
    print(f"✅ 可见记录数: {observed_records} ({observed_ratio:.2f}%)")
    print(f"✅ 不可见记录数: {total_records - observed_records} ({100-observed_ratio:.2f}%)")
    
    # =================== Step 5: 保存结果 ===================
    print(f"\n💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 保存
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 转换完成！结果已保存至: {output_csv_path}")
    
    # 显示每个节点的可见性统计
    print("\n📊 各节点可见性统计:")
    node_stats = results_df.groupby('node_id')['is_observed'].agg(['sum', 'count'])
    node_stats['ratio'] = node_stats['sum'] / node_stats['count'] * 100
    node_stats = node_stats.sort_values('ratio', ascending=False)
    
    for node_id, row in node_stats.iterrows():
        print(f"  {node_id}: {int(row['sum'])}/{int(row['count'])} ({row['ratio']:.1f}%)")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # 示例路径（请根据实际情况修改）
    # INPUT_CSV = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\minhang_lane_node_stats\0129094705_0001_node_patrol_mask.csv"  # 输入：节点级别的时间段mask
    # GRAPH_JSON_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\data\road_graph\minhang_graph.json"  # 图结构文件
    # OUTPUT_CSV = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\minhang_lane_node_stats\0129094705_0001_node_time_mask.csv"  # 输出：时间步级别的mask
    
    INPUT_CSV = '../data/lane_node_stats/d210291000_node_mask_5.csv'  # 输入：节点级别的时间段mask
    GRAPH_JSON_PATH = '../data/road_graph/graph_10m.json'  # 图结构文件
    OUTPUT_CSV = '../data/lane_node_stats/d210291000_node_mask_5.csv'  # 输出：时间步级别的mask

    # 检查文件是否存在
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 输入文件不存在: {INPUT_CSV}")
        print("请修改 INPUT_CSV 为实际的输入文件路径")
    elif not os.path.exists(GRAPH_JSON_PATH):
        print(f"❌ 图文件不存在: {GRAPH_JSON_PATH}")
        print("请修改 GRAPH_JSON_PATH 为实际的图文件路径")
    else:
        # 执行转换
        convert_node_mask_to_time_mask(INPUT_CSV, GRAPH_JSON_PATH, OUTPUT_CSV)

