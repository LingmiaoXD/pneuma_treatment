# -*- coding: utf-8 -*-
"""
11merge_time_periods.py

合并两个连续时间段的数据文件
- 合并 lane_mask 文件
- 合并 lane_node_stats 文件

自动计算第一个文件的最大时间，将第二个文件接续在其后
虽然两个文件的时间都是从0开始，但它们的绝对时间是连续的
"""

import pandas as pd
import os


def merge_lane_mask_files(file1_path, file2_path, output_path):
    """
    合并两个 lane_mask 文件
    
    参数:
        file1_path: str, 第一个时间段文件路径（8:30-9:00）
        file2_path: str, 第二个时间段文件路径（9:00-9:30）
        output_path: str, 输出文件路径
    """
    print(f"📦 正在读取第一个文件: {file1_path}")
    df1 = pd.read_csv(file1_path)
    print(f"✅ 第一个文件包含 {len(df1)} 条记录")
    print(f"   时间范围: {df1['start_frame'].min():.1f} - {df1['start_frame'].max():.1f}")
    
    print(f"📦 正在读取第二个文件: {file2_path}")
    df2 = pd.read_csv(file2_path)
    print(f"✅ 第二个文件包含 {len(df2)} 条记录")
    print(f"   原始时间范围: {df2['start_frame'].min():.1f} - {df2['start_frame'].max():.1f}")
    
    # 计算第一个文件的最大时间和时间窗口大小
    max_time_file1 = df1['start_frame'].max()
    # 获取第一个文件的所有唯一时间值，计算时间窗口大小
    unique_times_file1 = sorted(df1['start_frame'].unique())
    if len(unique_times_file1) > 1:
        time_window = unique_times_file1[1] - unique_times_file1[0]
    else:
        # 如果只有一个时间点，尝试从第二个文件获取时间窗口大小
        unique_times_file2 = sorted(df2['start_frame'].unique())
        if len(unique_times_file2) > 1:
            time_window = unique_times_file2[1] - unique_times_file2[0]
        else:
            time_window = 10.0  # 默认10（假设是10秒窗口）
    
    # 计算偏移量：第一个文件的最大时间 + 时间窗口大小
    time_offset = max_time_file1 + time_window
    print(f"   第一个文件最大时间: {max_time_file1:.1f}")
    print(f"   时间窗口大小: {time_window:.1f}")
    print(f"   计算的时间偏移量: {time_offset:.1f}")
    
    # 将第二个文件的时间加上偏移量
    df2['start_frame'] = df2['start_frame'] + time_offset
    print(f"   调整后时间范围: {df2['start_frame'].min():.1f} - {df2['start_frame'].max():.1f}")
    
    # 合并两个DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 按 start_frame 和 lane_id 排序
    merged_df = merged_df.sort_values(['start_frame', 'lane_id']).reset_index(drop=True)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"🎉 合并完成！")
    print(f"   总记录数: {len(merged_df)}")
    print(f"   时间范围: {merged_df['start_frame'].min():.1f} - {merged_df['start_frame'].max():.1f}")
    print(f"   已保存到: {output_path}")
    
    return merged_df


def merge_lane_node_stats_files(file1_path, file2_path, output_path):
    """
    合并两个 lane_node_stats 文件
    
    参数:
        file1_path: str, 第一个时间段文件路径（8:30-9:00）
        file2_path: str, 第二个时间段文件路径（9:00-9:30）
        output_path: str, 输出文件路径
    """
    print(f"📦 正在读取第一个文件: {file1_path}")
    df1 = pd.read_csv(file1_path)
    print(f"✅ 第一个文件包含 {len(df1)} 条记录")
    print(f"   时间范围: {df1['start_frame'].min():.1f} - {df1['start_frame'].max():.1f}")
    
    print(f"📦 正在读取第二个文件: {file2_path}")
    df2 = pd.read_csv(file2_path)
    print(f"✅ 第二个文件包含 {len(df2)} 条记录")
    print(f"   原始时间范围: {df2['start_frame'].min():.1f} - {df2['start_frame'].max():.1f}")
    
    # 计算第一个文件的最大时间和时间窗口大小
    max_time_file1 = df1['start_frame'].max()
    # 获取第一个文件的所有唯一时间值，计算时间窗口大小
    unique_times_file1 = sorted(df1['start_frame'].unique())
    if len(unique_times_file1) > 1:
        time_window = unique_times_file1[1] - unique_times_file1[0]
    else:
        # 如果只有一个时间点，尝试从第二个文件获取时间窗口大小
        unique_times_file2 = sorted(df2['start_frame'].unique())
        if len(unique_times_file2) > 1:
            time_window = unique_times_file2[1] - unique_times_file2[0]
        else:
            time_window = 10.0  # 默认10（假设是10秒窗口）
    
    # 计算偏移量：第一个文件的最大时间 + 时间窗口大小
    time_offset = max_time_file1 + time_window
    print(f"   第一个文件最大时间: {max_time_file1:.1f}")
    print(f"   时间窗口大小: {time_window:.1f}")
    print(f"   计算的时间偏移量: {time_offset:.1f}")
    
    # 将第二个文件的时间加上偏移量
    df2['start_frame'] = df2['start_frame'] + time_offset
    print(f"   调整后时间范围: {df2['start_frame'].min():.1f} - {df2['start_frame'].max():.1f}")
    
    # 合并两个DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 按 lane_id 和 start_frame 排序
    merged_df = merged_df.sort_values(['lane_id', 'start_frame']).reset_index(drop=True)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"🎉 合并完成！")
    print(f"   总记录数: {len(merged_df)}")
    print(f"   时间范围: {merged_df['start_frame'].min():.1f} - {merged_df['start_frame'].max():.1f}")
    print(f"   已保存到: {output_path}")
    
    return merged_df


def main():
    """
    主函数：合并两个时间段的数据文件
    """
    # =================== 配置参数 ===================
    # 第一个时间段
    MASK_FILE1 = r"../data/lane_node_stats/d210240830_lane_mask.csv"
    STATS_FILE1 = r"../data/lane_node_stats/d210240830_lane_node_stats.csv"
    
    # 第二个时间段
    MASK_FILE2 = r"../data/lane_node_stats/d210240900_lane_mask.csv"
    STATS_FILE2 = r"../data/lane_node_stats/d210240900_lane_node_stats.csv"
    
    # 输出文件路径
    MERGED_MASK_OUTPUT = r"../data/lane_node_stats/d210240900_merged_lane_mask.csv"
    MERGED_STATS_OUTPUT = r"../data/lane_node_stats/d210240900_merged_lane_node_stats.csv"
    
    # =================== 合并 lane_mask 文件 ===================
    print("=" * 60)
    print("🔄 开始合并 lane_mask 文件...")
    print("=" * 60)
    
    if not os.path.exists(MASK_FILE1):
        print(f"❌ 文件不存在: {MASK_FILE1}")
        return
    
    if not os.path.exists(MASK_FILE2):
        print(f"❌ 文件不存在: {MASK_FILE2}")
        return
    
    merge_lane_mask_files(MASK_FILE1, MASK_FILE2, MERGED_MASK_OUTPUT)
    
    # =================== 合并 lane_node_stats 文件 ===================
    print("\n" + "=" * 60)
    print("🔄 开始合并 lane_node_stats 文件...")
    print("=" * 60)
    
    if not os.path.exists(STATS_FILE1):
        print(f"❌ 文件不存在: {STATS_FILE1}")
        return
    
    if not os.path.exists(STATS_FILE2):
        print(f"❌ 文件不存在: {STATS_FILE2}")
        return
    
    merge_lane_node_stats_files(STATS_FILE1, STATS_FILE2, MERGED_STATS_OUTPUT)
    
    print("\n" + "=" * 60)
    print("✅ 所有文件合并完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

