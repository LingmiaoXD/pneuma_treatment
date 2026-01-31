# -*- coding: utf-8 -*-
"""
13test_data.py

根据node_mask过滤lane_node_stats，生成测试数据
根据配置参数决定保留盲点区域还是可见点区域的节点

输入：
- lane_node_stats CSV（来自09lane_node.py），包含完整的车道段统计数据
- node_mask CSV（格式：node_id,start,end），每一行表示当前node_id可见的连续区间

输出：
- 测试数据CSV，格式与lane_node_stats完全一样，根据配置参数只包含盲点或可见点区域的节点
- 如果remove_filtered=False，则保留所有行，被过滤的行属性字段为空值
"""

import os
import pandas as pd


def generate_test_data(lane_node_stats_path, NODE_MASK_PATH, output_path, keep_observed=0, remove_filtered=True):
    """
    根据node_mask过滤lane_node_stats，生成测试数据
    根据keep_observed参数决定保留盲点区域还是可见点区域的数据
    
    参数:
        lane_node_stats_path: str, lane_node_stats CSV文件路径
        NODE_MASK_PATH: str, node_mask CSV文件路径（格式：node_id,start,end）
        output_path: str, 输出CSV文件路径
        keep_observed: int, 保留类型：0表示保留盲点区域，1表示保留可见点区域
        remove_filtered: bool, 是否删除被过滤的行：
            - True（默认）: 直接删除被过滤的行，输出只包含符合条件的数据
            - False: 保留所有行，被过滤的行node_id和start_frame保留，其他属性字段设为空值
    """
    print("🚀 开始生成测试数据...")
    
    # =================== Step 1: 读取数据 ===================
    print("📦 正在读取lane_node_stats数据...")
    stats_df = pd.read_csv(lane_node_stats_path)
    
    # 检查必要字段
    required_fields = ['node_id', 'start_frame']
    missing_fields = [f for f in required_fields if f not in stats_df.columns]
    if missing_fields:
        raise ValueError(f"❌ lane_node_stats缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(stats_df)} 条统计记录")
    print(f"📊 涉及车道段数: {stats_df['node_id'].nunique()}")
    print(f"📊 时间窗口数: {stats_df['start_frame'].nunique()}")
    
    print("📦 正在读取node_mask数据...")
    mask_df = pd.read_csv(NODE_MASK_PATH)
    
    # 检查必要字段
    required_mask_fields = ['node_id', 'start', 'end']
    missing_mask_fields = [f for f in required_mask_fields if f not in mask_df.columns]
    if missing_mask_fields:
        raise ValueError(f"❌ node_mask缺少必要字段: {missing_mask_fields}")
    
    print(f"✅ 共读取 {len(mask_df)} 条可见区间记录")
    
    # =================== Step 2: 数据预处理 ===================
    # 确保数据类型一致
    stats_df['node_id'] = stats_df['node_id'].astype(str)
    stats_df['start_frame'] = stats_df['start_frame'].astype(int)
    
    mask_df['node_id'] = mask_df['node_id'].astype(str)
    mask_df['start'] = mask_df['start'].astype(int)
    mask_df['end'] = mask_df['end'].astype(int)
    
    # =================== Step 3: 根据可见区间判断每条记录是否可见 ===================
    if keep_observed == 0:
        print("🔄 正在过滤盲点区域的节点...")
    else:
        print("🔄 正在过滤可见点区域的节点...")
    
    # 按照node_id和start_frame排序
    stats_df = stats_df.sort_values(['node_id', 'start_frame']).reset_index(drop=True)
    
    # 为每条记录判断是否在可见区间内（使用绝对start_frame值）
    def is_in_visible_range(row):
        node_id = row['node_id']
        start_frame = row['start_frame']
        
        # 获取该node_id的所有可见区间
        node_masks = mask_df[mask_df['node_id'] == node_id]
        
        # 判断start_frame是否在任何一个可见区间内（使用绝对值对齐）
        for _, mask_row in node_masks.iterrows():
            if mask_row['start'] <= start_frame <= mask_row['end']:
                return 1  # 可见
        return 0  # 不可见
    
    print("🔍 正在判断每条记录的可见性（使用绝对start_frame值对齐）...")
    stats_df['is_observed'] = stats_df.apply(is_in_visible_range, axis=1)
    
    # 获取需要保留数据的行和需要过滤的行
    keep_mask = stats_df['is_observed'] == keep_observed
    
    if remove_filtered:
        # 直接删除被过滤的行
        filtered_df = stats_df[keep_mask].copy()
        # 删除辅助列
        filtered_df = filtered_df.drop(columns=['is_observed'])
    else:
        # 保留所有行，但被过滤的行属性字段设为空值
        filtered_df = stats_df.copy()
        
        # 获取需要清空的属性列（除了node_id和start_frame之外的所有列）
        key_columns = ['node_id', 'start_frame', 'is_observed']
        attr_columns = [col for col in filtered_df.columns if col not in key_columns]
        
        # 将被过滤行的属性字段设为空值
        filtered_df.loc[~keep_mask, attr_columns] = None
        
        # 删除辅助列
        filtered_df = filtered_df.drop(columns=['is_observed'])
    
    # 按照原始顺序排序
    filtered_df = filtered_df.sort_values(['node_id', 'start_frame']).reset_index(drop=True)
    
    # =================== Step 4: 保存结果 ===================
    print(f"💾 正在保存结果到 {output_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存CSV文件
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    
    region_type = "盲点区域" if keep_observed == 0 else "可见点区域"
    kept_count = keep_mask.sum()
    filtered_count = (~keep_mask).sum()
    
    print(f"🎉 {region_type}测试数据已保存至: {output_path}")
    print(f"📊 原始记录数: {len(stats_df)}")
    print(f"📊 输出记录数: {len(filtered_df)}")
    print(f"📊 {region_type}记录数（有效数据）: {kept_count}")
    if not remove_filtered:
        print(f"📊 被过滤记录数（属性为空）: {filtered_count}")
    print(f"📊 {region_type}占比: {kept_count / len(stats_df):.2%}")
    print(f"📊 涉及车道段数: {filtered_df['node_id'].nunique()}")
    print(f"📊 时间窗口数: {filtered_df['start_frame'].nunique()}")
    
    return filtered_df


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # =================== 配置参数 ===================
    # 选择保留类型：0表示保留盲点区域（is_observed == 0），1表示保留可见点区域（is_observed == 1）
    KEEP_OBSERVED = 1  # 开发者可在此处修改：0或1
    
    # 是否删除被过滤的行：
    # - True: 直接删除被过滤的行，输出只包含符合条件的数据（与原逻辑一致）
    # - False: 保留所有行，被过滤的行node_id和start_frame保留，其他属性字段设为空值
    REMOVE_FILTERED = False  # 开发者可在此处修改：True或False
    
    LANE_NODE_STATS_PATH = r"../data/lane_node_stats/d210291000_lane_node_stats.csv"  # 完整的lane_node_stats
    NODE_MASK_PATH = r"../data/lane_node_stats/d210291000_node_mask.csv"  # node_mask（格式：node_id,start,end）
    OUTPUT_CSV = r"../data/lane_node_stats/d210291000_test_data.csv"  # 输出路径（根据KEEP_OBSERVED决定保留盲点或可见点数据）
    
    # 检查文件是否存在
    if not os.path.exists(LANE_NODE_STATS_PATH):
        raise FileNotFoundError(f"❌ lane_node_stats文件不存在: {LANE_NODE_STATS_PATH}")
    
    if not os.path.exists(NODE_MASK_PATH):
        raise FileNotFoundError(f"❌ lane_mask文件不存在: {NODE_MASK_PATH}")
    
    # 执行生成
    generate_test_data(LANE_NODE_STATS_PATH, NODE_MASK_PATH, OUTPUT_CSV, keep_observed=KEEP_OBSERVED, remove_filtered=REMOVE_FILTERED)





