# -*- coding: utf-8 -*-
"""
11merge_realworld_with_mask.py

根据节点级mask文件，将不可见时段的dynamic数据置空

输入：
    1. 节点级mask文件，有三个属性字段node_id，start，end，每一行表示在start到end时间内，可见node_id
       （是的，虽然它叫mask，但记录的都是可见的时间段）
    2. dynamic文件，参考d210240930_lane_node_stats.csv

处理：
    对于dynamic文件的每一行，如果当前这一行根据时间和node_id，根本不可见，
    就把这一行的avg_speed,avg_occupancy,total_vehicles三个值都置空（不是0，是置空）

输出：
    修改后的dynamic文件
"""

import os
import pandas as pd
import numpy as np


def load_mask_data(mask_csv_path):
    """
    加载mask文件，构建可见性查询结构
    
    参数:
        mask_csv_path: str, mask文件路径
        
    返回:
        dict: {node_id: [(start, end), ...]} 每个节点的可见时间段列表
    """
    print(f"📦 正在读取mask文件: {mask_csv_path}")
    mask_df = pd.read_csv(mask_csv_path)
    
    # 检查必要字段
    required_fields = ['node_id', 'start', 'end']
    missing_fields = [f for f in required_fields if f not in mask_df.columns]
    if missing_fields:
        raise ValueError(f"❌ Mask文件缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(mask_df)} 条mask记录")
    
    # 构建 node_id -> [(start, end), ...] 的映射
    visibility_dict = {}
    for _, row in mask_df.iterrows():
        node_id = int(row['node_id'])
        start = row['start']
        end = row['end']
        
        if node_id not in visibility_dict:
            visibility_dict[node_id] = []
        visibility_dict[node_id].append((start, end))
    
    print(f"✅ 共有 {len(visibility_dict)} 个节点有可见性记录")
    
    return visibility_dict


def is_visible(node_id, time_frame, visibility_dict):
    """
    判断某个节点在某个时间是否可见
    
    参数:
        node_id: int, 节点ID
        time_frame: float, 时间帧
        visibility_dict: dict, 可见性字典
        
    返回:
        bool: True表示可见，False表示不可见
    """
    if node_id not in visibility_dict:
        # 如果节点不在mask中，认为不可见
        return False
    
    # 检查时间是否在任何一个可见时间段内
    for start, end in visibility_dict[node_id]:
        if start <= time_frame < end:
            return True
    
    return False


def main(mask_csv_path, dynamic_csv_path, output_csv_path):
    """
    主函数
    
    参数:
        mask_csv_path: str, mask文件路径
        dynamic_csv_path: str, dynamic文件路径
        output_csv_path: str, 输出文件路径
    """
    print("🚀 开始合并mask和dynamic数据...")
    
    # =================== Step 1: 读取数据 ===================
    # 加载mask数据
    visibility_dict = load_mask_data(mask_csv_path)
    
    # 加载dynamic数据
    print(f"\n📦 正在读取dynamic文件: {dynamic_csv_path}")
    dynamic_df = pd.read_csv(dynamic_csv_path)
    
    # 检查必要字段
    required_fields = ['node_id', 'start_frame', 'avg_speed', 'avg_occupancy', 'total_vehicles']
    missing_fields = [f for f in required_fields if f not in dynamic_df.columns]
    if missing_fields:
        raise ValueError(f"❌ Dynamic文件缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(dynamic_df)} 条dynamic记录")
    
    # =================== Step 2: 处理数据 ===================
    print("\n📊 正在处理数据...")
    
    # 统计信息
    total_rows = len(dynamic_df)
    masked_rows = 0
    
    # 遍历每一行，检查可见性
    for idx, row in dynamic_df.iterrows():
        node_id = int(row['node_id'])
        time_frame = row['start_frame']
        
        # 判断是否可见
        if not is_visible(node_id, time_frame, visibility_dict):
            # 不可见，将三个字段置空
            dynamic_df.at[idx, 'avg_speed'] = np.nan
            dynamic_df.at[idx, 'avg_occupancy'] = np.nan
            dynamic_df.at[idx, 'total_vehicles'] = np.nan
            masked_rows += 1
    
    print(f"✅ 处理完成")
    print(f"📊 总记录数: {total_rows}")
    print(f"📊 被置空记录数: {masked_rows} ({masked_rows/total_rows*100:.2f}%)")
    print(f"📊 保留记录数: {total_rows - masked_rows} ({(total_rows-masked_rows)/total_rows*100:.2f}%)")
    
    # =================== Step 3: 保存结果 ===================
    print(f"\n💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 保存结果
    dynamic_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 结果已保存至: {output_csv_path}")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # 示例路径（请根据实际情况修改）
    MASK_CSV_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\minhang_lane_node_stats\k0129094705_0001_node_mask.csv"  # mask文件
    DYNAMIC_CSV_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\minhang_lane_node_stats\k0129094705_0001_lane_node_stats_3.csv"  # dynamic文件
    OUTPUT_CSV = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\minhang_lane_node_stats\k0129094705_0001_lane_node_stats.csv"  # 输出路径
    
    # 检查文件是否存在
    if not os.path.exists(MASK_CSV_PATH):
        print(f"❌ Mask文件不存在: {MASK_CSV_PATH}")
        print("请修改 MASK_CSV_PATH 为实际的mask文件路径")
    elif not os.path.exists(DYNAMIC_CSV_PATH):
        print(f"❌ Dynamic文件不存在: {DYNAMIC_CSV_PATH}")
        print("请修改 DYNAMIC_CSV_PATH 为实际的dynamic文件路径")
    else:
        # 执行处理
        main(MASK_CSV_PATH, DYNAMIC_CSV_PATH, OUTPUT_CSV)