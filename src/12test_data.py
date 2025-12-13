# -*- coding: utf-8 -*-
"""
12test_data.py

根据lane_mask过滤lane_node_stats，生成测试数据
根据配置参数决定保留盲点区域（is_observed == 0）还是可见点区域（is_observed == 1）的节点

输入：
- lane_node_stats CSV（来自09lane_node.py），包含完整的车道段统计数据
- lane_mask CSV（来自10mask.py），包含每个车道段在每个时间窗口的观测状态

输出：
- 测试数据CSV，格式与lane_node_stats完全一样，根据配置参数只包含盲点或可见点区域的节点
"""

import os
import pandas as pd


def generate_test_data(lane_node_stats_path, lane_mask_path, output_path, keep_observed=0):
    """
    根据lane_mask过滤lane_node_stats，生成测试数据
    根据keep_observed参数决定保留盲点区域（is_observed == 0）还是可见点区域（is_observed == 1）的数据
    
    参数:
        lane_node_stats_path: str, lane_node_stats CSV文件路径
        lane_mask_path: str, lane_mask CSV文件路径
        output_path: str, 输出CSV文件路径
        keep_observed: int, 保留类型：0表示保留盲点区域（is_observed == 0），1表示保留可见点区域（is_observed == 1）
    """
    print("🚀 开始生成测试数据...")
    
    # =================== Step 1: 读取数据 ===================
    print("📦 正在读取lane_node_stats数据...")
    stats_df = pd.read_csv(lane_node_stats_path)
    
    # 检查必要字段
    required_fields = ['lane_id', 'start_frame']
    missing_fields = [f for f in required_fields if f not in stats_df.columns]
    if missing_fields:
        raise ValueError(f"❌ lane_node_stats缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(stats_df)} 条统计记录")
    print(f"📊 涉及车道段数: {stats_df['lane_id'].nunique()}")
    print(f"📊 时间窗口数: {stats_df['start_frame'].nunique()}")
    
    print("📦 正在读取lane_mask数据...")
    mask_df = pd.read_csv(lane_mask_path)
    
    # 检查必要字段
    required_mask_fields = ['lane_id', 'start_frame', 'is_observed']
    missing_mask_fields = [f for f in required_mask_fields if f not in mask_df.columns]
    if missing_mask_fields:
        raise ValueError(f"❌ lane_mask缺少必要字段: {missing_mask_fields}")
    
    print(f"✅ 共读取 {len(mask_df)} 条掩码记录")
    
    # =================== Step 2: 数据预处理 ===================
    # 确保数据类型一致
    stats_df['lane_id'] = stats_df['lane_id'].astype(int)
    stats_df['start_frame'] = stats_df['start_frame'].astype(float)
    
    mask_df['lane_id'] = mask_df['lane_id'].astype(int)
    mask_df['start_frame'] = mask_df['start_frame'].astype(float)
    mask_df['is_observed'] = mask_df['is_observed'].astype(int)
    
    # =================== Step 3: 合并数据并过滤 ===================
    if keep_observed == 0:
        print("🔄 正在合并数据并过滤盲点区域的节点...")
    else:
        print("🔄 正在合并数据并过滤可见点区域的节点...")
    
    # 将stats_df和mask_df合并，基于lane_id和start_frame
    merged_df = stats_df.merge(
        mask_df[['lane_id', 'start_frame', 'is_observed']],
        on=['lane_id', 'start_frame'],
        how='inner'
    )
    
    # 根据keep_observed参数决定保留哪种类型的记录
    filtered_df = merged_df[merged_df['is_observed'] == keep_observed].copy()
    
    # 删除is_observed列（因为输出格式要与lane_node_stats完全一样）
    filtered_df = filtered_df.drop(columns=['is_observed'])
    
    # 按照原始顺序排序
    filtered_df = filtered_df.sort_values(['lane_id', 'start_frame']).reset_index(drop=True)
    
    # =================== Step 4: 保存结果 ===================
    print(f"💾 正在保存结果到 {output_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存CSV文件
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    
    region_type = "盲点区域" if keep_observed == 0 else "可见点区域"
    print(f"🎉 {region_type}测试数据已保存至: {output_path}")
    print(f"📊 原始记录数: {len(stats_df)}")
    print(f"📊 {region_type}记录数: {len(filtered_df)}")
    print(f"📊 {region_type}占比: {len(filtered_df) / len(stats_df):.2%}")
    print(f"📊 涉及车道段数: {filtered_df['lane_id'].nunique()}")
    print(f"📊 时间窗口数: {filtered_df['start_frame'].nunique()}")
    
    return filtered_df


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # =================== 配置参数 ===================
    # 选择保留类型：0表示保留盲点区域（is_observed == 0），1表示保留可见点区域（is_observed == 1）
    KEEP_OBSERVED = 0  # 开发者可在此处修改：0或1
    
    LANE_NODE_STATS_PATH = r"../data/lane_node_stats/d210291000_lane_node_stats.csv"  # 完整的lane_node_stats
    LANE_MASK_PATH = r"../data/lane_node_stats/d210291000_lane_mask.csv"  # lane_mask
    OUTPUT_CSV = r"../data/lane_node_stats/d210291000_test_data.csv"  # 输出路径（根据KEEP_OBSERVED决定保留盲点或可见点数据）
    
    # 检查文件是否存在
    if not os.path.exists(LANE_NODE_STATS_PATH):
        raise FileNotFoundError(f"❌ lane_node_stats文件不存在: {LANE_NODE_STATS_PATH}")
    
    if not os.path.exists(LANE_MASK_PATH):
        raise FileNotFoundError(f"❌ lane_mask文件不存在: {LANE_MASK_PATH}")
    
    # 执行生成
    generate_test_data(LANE_NODE_STATS_PATH, LANE_MASK_PATH, OUTPUT_CSV, keep_observed=KEEP_OBSERVED)





