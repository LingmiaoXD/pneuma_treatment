# -*- coding: utf-8 -*-
"""
stat_lane_transitions.py

统计轨迹数据中每个车道段到下一个车道段的ID变动情况

输出格式：
- CSV文件，包含 from_lane_id, to_lane_id, count 等字段
- JSON文件，包含 lanes 信息（lane_id, nodes, stopline_node, segment_length, total_length）
"""

import os
import json
import pandas as pd
from collections import Counter, defaultdict


def main(traj_csv_path, output_csv_path, output_json_path=None, segment_length=40.0):
    """
    主函数

    参数:
        traj_csv_path: str, 轨迹 CSV 路径，需包含 id, frame, FID, lane_id 等字段
        output_csv_path: str, 输出 CSV 文件路径
        output_json_path: str, 输出 JSON 文件路径（可选）
        segment_length: float, 每个节点的段长度，用于计算 total_length
    """
    print("🚀 开始统计车道段ID变动情况...")

    # =================== Step 1: 读取轨迹数据 ===================
    print("📦 正在读取轨迹数据...")
    traj_df = pd.read_csv(traj_csv_path)
    
    # 检查必要字段
    required_fields = ['id', 'frame', 'FID', 'lane_id']
    missing_fields = [f for f in required_fields if f not in traj_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 轨迹数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(traj_df)} 条轨迹记录")
    
    # 处理frame字段（如果有分号）
    if 'frame' in traj_df.columns:
        traj_df['frame'] = traj_df['frame'].astype(str).str.rstrip(';')
        traj_df['frame'] = traj_df['frame'].astype(float)
    
    # 过滤掉没有车道段ID的记录（包括NaN和空字符串）
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna()].copy()
    # 同时过滤掉空字符串
    traj_df = traj_df[traj_df['FID'].astype(str).str.strip() != ''].copy()
    filtered_count = len(traj_df)
    print(f"📊 过滤后保留 {filtered_count} 条有效记录（过滤前: {original_count}）")

    # =================== Step 2: 按车辆ID和frame排序 ===================
    print("🔄 正在排序轨迹数据...")
    traj_df = traj_df.sort_values(["id", "frame"]).copy()
    
    # 确保车道段ID为字符串类型，便于统计
    traj_df['FID'] = traj_df['FID'].astype(str).str.strip()
    
    # =================== Step 3: 提取车道段变动 ===================
    print("🔍 正在提取车道段ID变动...")
    
    def extract_lane_transitions(group):
        """从单个车辆的轨迹中提取车道段变动"""
        transitions = []
        prev_lane_id = None
        
        for _, row in group.iterrows():
            # FID字段存储的是车道段的id值（已经是字符串类型）
            curr_lane_id = row["FID"]
            
            # 如果当前车道段ID与上一个不同，记录一次变动
            if prev_lane_id is not None and prev_lane_id != curr_lane_id:
                transitions.append((prev_lane_id, curr_lane_id))
            
            prev_lane_id = curr_lane_id
        
        return transitions

    # 按车辆ID分组，提取每个车辆的车道段变动
    all_transitions = []
    for vehicle_id, group in traj_df.groupby("id"):
        transitions = extract_lane_transitions(group)
        all_transitions.extend(transitions)
    
    print(f"✅ 共提取 {len(all_transitions)} 次车道段变动")

    # =================== Step 4: 统计变动频次 ===================
    print("📊 正在统计变动频次...")
    transition_counter = Counter(all_transitions)
    
    # 转换为DataFrame
    transition_data = []
    for (from_lane_id, to_lane_id), count in transition_counter.items():
        transition_data.append({
            'from_lane_id': from_lane_id,
            'to_lane_id': to_lane_id,
            'count': count
        })
    
    transition_df = pd.DataFrame(transition_data)
    
    # 按频次降序排序
    transition_df = transition_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"✅ 共统计到 {len(transition_df)} 种不同的车道段变动组合")

    # =================== Step 5: 输出统计结果 ===================
    print(f"💾 正在保存统计结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 保存CSV
    transition_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 统计结果已保存至: {output_csv_path}")
    print(f"📊 总计变动类型数: {len(transition_df)}")
    
    # =================== Step 6: 生成 lanes JSON ===================
    if output_json_path:
        print(f"📝 正在生成 lanes JSON...")
        
        # 构建 lane_id -> nodes (FID列表) 的映射
        lane_nodes_map = defaultdict(set)
        for _, row in traj_df.iterrows():
            lane_id = row['lane_id']
            fid = row['FID']
            if pd.notna(lane_id) and pd.notna(fid):
                # 尝试转换为整数，如果失败则保持原值
                try:
                    lane_id_val = int(float(lane_id))
                except (ValueError, TypeError):
                    lane_id_val = lane_id
                try:
                    fid_val = int(float(fid))
                except (ValueError, TypeError):
                    fid_val = fid
                lane_nodes_map[lane_id_val].add(fid_val)
        
        # 构建 lanes 列表
        lanes_list = []
        for lane_id in sorted(lane_nodes_map.keys()):
            nodes = sorted(list(lane_nodes_map[lane_id]))
            num_nodes = len(nodes)
            total_length = segment_length * num_nodes
            
            lane_info = {
                "lane_id": lane_id,
                "nodes": nodes,
                "stopline_node": None,  # 先空着，后续手动填写
                "total_length": total_length,
                "segment_length": segment_length
            }
            lanes_list.append(lane_info)
        
        # 输出 JSON
        output_data = {"lanes": lanes_list}
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"🎉 lanes JSON 已保存至: {output_json_path}")
        print(f"📊 总计 {len(lanes_list)} 条车道信息")
    


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    TRAJ_CSV_PATH = r"../data/trajectory_with_laneid/d210291000.csv"         # 轨迹数据，需包含 id, frame, FID, lane_id 等字段
    OUTPUT_CSV = r"../data/road_graph/d210291000_transitions.csv"      # 输出路径
    OUTPUT_JSON = r"../data/road_graph/d210291000_lanes.json"          # JSON 输出路径
    SEGMENT_LENGTH = 10.0  # 每个节点的段长度，可根据需要调整

    # 执行统计
    main(TRAJ_CSV_PATH, OUTPUT_CSV, OUTPUT_JSON, SEGMENT_LENGTH)

