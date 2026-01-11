# -*- coding: utf-8 -*-
"""
11merge_time_only.py

合并多个连续时间段的 lane_node_stats 文件
- 支持多个文件按顺序依次合并
- 自动计算时间偏移量，保证时间连续
- 打印每个文件在最终输出中的时间范围
"""

import pandas as pd
import os


def merge_multiple_lane_node_stats(file_paths, output_path):
    """
    合并多个 lane_node_stats 文件
    
    参数:
        file_paths: list[str], 按时间顺序排列的文件路径列表
        output_path: str, 输出文件路径
    
    返回:
        merged_df: DataFrame, 合并后的数据
    """
    if len(file_paths) == 0:
        print("❌ 没有提供任何文件")
        return None
    
    if len(file_paths) == 1:
        print("⚠️ 只有一个文件，直接复制")
        df = pd.read_csv(file_paths[0])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        return df
    
    # 存储每个文件的信息
    file_info = []
    merged_dfs = []
    current_offset = 0.0
    time_window = None
    
    print("=" * 70)
    print("🔄 开始合并 lane_node_stats 文件...")
    print("=" * 70)
    
    for i, file_path in enumerate(file_paths):
        print(f"\n📦 [{i+1}/{len(file_paths)}] 正在处理: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        original_min = df['start_frame'].min()
        original_max = df['start_frame'].max()
        print(f"   原始时间范围: {original_min:.1f} - {original_max:.1f}")
        print(f"   记录数: {len(df)}")
        
        # 计算时间窗口大小（从第一个文件获取）
        if time_window is None:
            unique_times = sorted(df['start_frame'].unique())
            if len(unique_times) > 1:
                time_window = unique_times[1] - unique_times[0]
            else:
                time_window = 10.0  # 默认10秒
            print(f"   检测到时间窗口大小: {time_window:.1f}")
        
        # 应用时间偏移
        if i == 0:
            # 第一个文件不需要偏移
            adjusted_min = original_min
            adjusted_max = original_max
        else:
            # 后续文件需要偏移
            df['start_frame'] = df['start_frame'] + current_offset
            adjusted_min = df['start_frame'].min()
            adjusted_max = df['start_frame'].max()
            print(f"   应用偏移量: {current_offset:.1f}")
            print(f"   调整后时间范围: {adjusted_min:.1f} - {adjusted_max:.1f}")
        
        # 记录文件信息
        file_info.append({
            'file': os.path.basename(file_path),
            'original_range': f"{original_min:.1f} - {original_max:.1f}",
            'merged_range_start': adjusted_min,
            'merged_range_end': adjusted_max
        })
        
        merged_dfs.append(df)
        
        # 更新偏移量：当前文件的最大时间 + 时间窗口
        current_offset = adjusted_max + time_window
    
    # 合并所有DataFrame
    print("\n" + "-" * 70)
    print("🔗 正在合并所有文件...")
    merged_df = pd.concat(merged_dfs, ignore_index=True)
    
    # 按 node_id 和 start_frame 排序
    merged_df = merged_df.sort_values(['node_id', 'start_frame']).reset_index(drop=True)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 打印汇总信息
    print("\n" + "=" * 70)
    print("📊 合并结果汇总")
    print("=" * 70)
    print(f"{'序号':<4} {'文件名':<40} {'在合并文件中的时间范围':<25}")
    print("-" * 70)
    for i, info in enumerate(file_info):
        range_str = f"{info['merged_range_start']:.1f} - {info['merged_range_end']:.1f}"
        print(f"{i+1:<4} {info['file']:<40} {range_str:<25}")
    print("-" * 70)
    print(f"总记录数: {len(merged_df)}")
    print(f"总时间范围: {merged_df['start_frame'].min():.1f} - {merged_df['start_frame'].max():.1f}")
    print(f"输出文件: {output_path}")
    print("=" * 70)
    print("✅ 合并完成！")
    
    return merged_df


def main():
    """
    主函数：合并多个时间段的 lane_node_stats 文件
    """
    # =================== 配置参数 ===================
    # 按时间顺序排列的文件列表
    STATS_FILES = [
        r"../data/lane_node_stats/d210240830_lane_node_stats.csv",
        r"../data/lane_node_stats/d210240900_lane_node_stats.csv",
        r"../data/lane_node_stats/d210240930_lane_node_stats.csv",
    ]
    
    # 输出文件路径
    OUTPUT_FILE = r"../data/lane_node_stats/merged_lane_node_stats.csv"
    
    # =================== 执行合并 ===================
    merge_multiple_lane_node_stats(STATS_FILES, OUTPUT_FILE)


if __name__ == "__main__":
    main()
