# -*- coding: utf-8 -*-
"""
09lane_node_NumStatic.py

按照节点ID(node_id)统计总共经过的车辆个数

输入：
- 轨迹CSV（来自05trajectory_with_laneid.py），包含 id, frame, FID(node_id) 等字段
- graph.json（道路图结构，包含 lanes 和 nodes）

输出：
- CSV文件，每行代表一个节点经过的总车辆数
"""

import os
import json
import pandas as pd


def load_graph(graph_json_path):
    """
    加载图结构，获取所有节点ID
    
    参数:
        graph_json_path: str, graph.json文件路径
        
    返回:
        set: 所有节点ID的集合
    """
    print(f"📦 正在读取图结构: {graph_json_path}")
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 获取所有节点ID
    node_ids = set()
    for node in graph_data.get('nodes', []):
        node_id = int(node['node_id'])
        node_ids.add(node_id)
    
    print(f"✅ 共加载 {len(node_ids)} 个节点")
    return node_ids


def main(traj_csv_path, graph_json_path, output_csv_path):
    """
    主函数：统计每个节点经过的车辆总数
    
    参数:
        traj_csv_path: str, 轨迹CSV路径
        graph_json_path: str, graph.json文件路径
        output_csv_path: str, 输出CSV路径
    """
    print("🚀 开始统计节点车辆总数...")
    
    # =================== Step 1: 读取数据 ===================
    print("📦 正在读取轨迹数据...")
    traj_df = pd.read_csv(traj_csv_path)
    
    # 检查必要字段
    required_fields = ['id', 'FID']
    missing_fields = [f for f in required_fields if f not in traj_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 轨迹数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(traj_df)} 条轨迹记录")
    
    # 过滤掉没有节点ID的记录
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna()].copy()
    traj_df = traj_df[traj_df['FID'].astype(str).str.strip() != ''].copy()
    filtered_count = len(traj_df)
    print(f"📊 过滤后保留 {filtered_count} 条有效记录（过滤前: {original_count}）")
    
    # 确保FID为整数类型
    traj_df['FID'] = traj_df['FID'].astype(str).str.strip()
    traj_df['FID_int'] = traj_df['FID'].apply(lambda x: int(float(x)) if x else -1)
    
    # 加载图结构
    node_ids = load_graph(graph_json_path)
    
    # =================== Step 2: 统计每个节点的车辆总数 ===================
    print("📊 正在统计每个节点经过的车辆总数...")
    
    results = []
    
    # 对每个节点统计唯一车辆数
    for node_id in node_ids:
        node_data = traj_df[traj_df['FID_int'] == node_id]
        
        # 统计唯一车辆数（不归一化）
        total_vehicles = node_data['id'].nunique() if not node_data.empty else 0
        
        results.append({
            'node_id': node_id,
            'total_vehicles': total_vehicles
        })
    
    # =================== Step 3: 保存结果 ===================
    print(f"💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('node_id').reset_index(drop=True)
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 统计结果已保存至: {output_csv_path}")
    print(f"📊 总计节点数: {len(results_df)}")
    print(f"📊 车辆总数: {traj_df['id'].nunique()}")
    print(f"📊 平均每节点车辆数: {results_df['total_vehicles'].mean():.2f}")
    print(f"📊 最大车辆数节点: {results_df['total_vehicles'].max()}")
    print(f"📊 最小车辆数节点: {results_df['total_vehicles'].min()}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    TRAJ_CSV_PATH = r"../data/trajectory_with_laneid/d210291000.csv"  # 轨迹数据
    GRAPH_JSON_PATH = r"../data/road_graph/graph_10m.json"  # 图结构（更新版本，包含lanes和nodes）
    OUTPUT_CSV = r"../data/lane_node_stats/d210291000_lane_node_num.csv"  # 输出路径
    
    if not os.path.exists(GRAPH_JSON_PATH):
        raise FileNotFoundError(f"❌ 图文件不存在: {GRAPH_JSON_PATH}")
    
    # 执行统计
    main(TRAJ_CSV_PATH, GRAPH_JSON_PATH, OUTPUT_CSV)

