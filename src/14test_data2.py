# -*- coding: utf-8 -*-
"""

根据lane_mask过滤lane_node_stats，生成测试数据
只保留标注为有数据的节点（is_observed == 1）

输入：
- lane_node_stats CSV（来自09lane_node.py），包含完整的车道段统计数据
- lane_mask CSV（来自10mask.py），包含每个车道段在每个时间窗口的观测状态

输出：
- 测试数据CSV，格式与lane_node_stats完全一样，但只包含有数据的节点
"""

import os
import numpy as np
import pandas as pd


def generate_test_data(lane_node_stats_path, lane_mask_path, output_path, noise_ratio=0.05):
    """
    根据lane_mask过滤lane_node_stats，生成测试数据
    
    参数:
        lane_node_stats_path: str, lane_node_stats CSV文件路径
        lane_mask_path: str, lane_mask CSV文件路径
        output_path: str, 输出CSV文件路径
        noise_ratio: float, 随机波动幅度比例（默认0.05，即5%的波动）
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
    print("🔄 正在合并数据并过滤有数据的节点...")
    
    # 将stats_df和mask_df合并，基于lane_id和start_frame
    merged_df = stats_df.merge(
        mask_df[['lane_id', 'start_frame', 'is_observed']],
        on=['lane_id', 'start_frame'],
        how='inner'
    )
    
    # 只保留is_observed == 1的记录
    filtered_df = merged_df[merged_df['is_observed'] == 1].copy()
    
    # 删除is_observed列（因为输出格式要与lane_node_stats完全一样）
    filtered_df = filtered_df.drop(columns=['is_observed'])
    
    # 按照原始顺序排序
    filtered_df = filtered_df.sort_values(['lane_id', 'start_frame']).reset_index(drop=True)
    
    # =================== Step 4: 添加随机波动 ===================
    print(f"🔀 正在为指标值添加随机波动（波动幅度: {noise_ratio*100:.1f}%）...")
    
    # 识别需要添加波动的列（排除lane_id和start_frame）
    exclude_columns = ['lane_id', 'start_frame']
    value_columns = [col for col in filtered_df.columns if col not in exclude_columns]
    
    # 对每个指标列添加随机波动
    for col in value_columns:
        if filtered_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # 保存原始数据类型
            original_dtype = filtered_df[col].dtype
            is_integer = original_dtype in [np.int64, np.int32]
            
            # 获取原始值（转换为浮点数以便计算）
            original_values = filtered_df[col].values.astype(float).copy()
            
            # 生成随机波动（均匀分布，范围在 [-noise_ratio, noise_ratio]）
            noise = np.random.uniform(-noise_ratio, noise_ratio, size=len(filtered_df))
            
            # 应用波动：新值 = 原值 * (1 + 波动)
            new_values = original_values * (1 + noise)
            
            # 处理特殊情况：
            # 1. 如果原值是-1（表示无效值），保持为-1
            # 2. 如果原值在[0, 1]范围内（可能是ratio类），限制结果在[0, 1]范围内
            # 3. 如果原值是非负的，确保结果也是非负的
            mask_negative_one = (original_values == -1)
            mask_ratio_range = (original_values >= 0) & (original_values <= 1) & (~mask_negative_one)
            mask_other_positive = (original_values > 1) & (~mask_negative_one)
            
            # 保持-1不变
            new_values[mask_negative_one] = -1
            
            # ratio类值限制在[0, 1]范围内
            if np.any(mask_ratio_range):
                new_values[mask_ratio_range] = np.clip(new_values[mask_ratio_range], 0.0, 1.0)
            
            # 其他正值确保非负
            if np.any(mask_other_positive):
                new_values[mask_other_positive] = np.maximum(new_values[mask_other_positive], 0.0)
            
            # 根据原始数据类型处理结果
            if is_integer:
                # 整数列：四舍五入并转换回整数类型
                filtered_df[col] = np.round(new_values).astype(original_dtype)
            else:
                # 浮点数列：保留2位小数（与原始数据格式一致）
                filtered_df[col] = np.round(new_values, 2)
    
    print(f"✅ 已为 {len(value_columns)} 个指标列添加随机波动")
    
    # =================== Step 5: 保存结果 ===================
    print(f"💾 正在保存结果到 {output_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存CSV文件
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"🎉 测试数据已保存至: {output_path}")
    print(f"📊 原始记录数: {len(stats_df)}")
    print(f"📊 过滤后记录数: {len(filtered_df)}")
    print(f"📊 数据减少比例: {1 - len(filtered_df) / len(stats_df):.2%}")
    print(f"📊 涉及车道段数: {filtered_df['lane_id'].nunique()}")
    print(f"📊 时间窗口数: {filtered_df['start_frame'].nunique()}")
    
    return filtered_df


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    LANE_NODE_STATS_PATH = r"../data/lane_node_stats/d210291000_lane_node_stats.csv"  # 完整的lane_node_stats
    LANE_MASK_PATH = r"../data/lane_node_stats/d210291000_lane_mask.csv"  # lane_mask
    OUTPUT_CSV = r"../data/lane_node_stats/d210291000_test_data.csv"  # 输出路径
    
    # =================== 波动参数配置 ===================
    # 随机波动幅度比例（0.05表示5%的波动，开发者可根据需要调整）
    NOISE_RATIO = 0.3
    
    # 检查文件是否存在
    if not os.path.exists(LANE_NODE_STATS_PATH):
        raise FileNotFoundError(f"❌ lane_node_stats文件不存在: {LANE_NODE_STATS_PATH}")
    
    if not os.path.exists(LANE_MASK_PATH):
        raise FileNotFoundError(f"❌ lane_mask文件不存在: {LANE_MASK_PATH}")
    
    # 执行生成
    generate_test_data(LANE_NODE_STATS_PATH, LANE_MASK_PATH, OUTPUT_CSV, noise_ratio=NOISE_RATIO)




