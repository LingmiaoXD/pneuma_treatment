# -*- coding: utf-8 -*-
"""
15draw_plot.py

根据指定的 start_frame 筛选 lane_node_stats 数据，
将统计数据合并到 buffer shapefile 中，生成新的 shapefile。

输入:
    - plots/buffer/buffer_small_crossing_3.shp: 基础 shapefile（包含 FID 字段）
    - data/lane_node_stats/d210291000_lane_node_stats.csv: 车道统计数据

输出:
    - plots/buffer/d210291000_buffer.shp: 合并后的 shapefile
"""

import os
import pandas as pd
import geopandas as gpd
from shapefile_utils import read_shapefile_with_fid


def create_buffer_with_stats(
    buffer_shp_path: str,
    stats_csv_path: str,
    output_shp_path: str,
    start_frame: float,
    verbose: bool = True
):
    """
    根据指定的 start_frame 筛选统计数据，合并到 buffer shapefile 中
    
    参数:
        buffer_shp_path: str, 基础 buffer shapefile 路径
        stats_csv_path: str, 车道统计数据 CSV 路径
        output_shp_path: str, 输出 shapefile 路径
        start_frame: float, 要筛选的 start_frame 值
        verbose: bool, 是否打印详细信息
    """
    # 1. 读取基础 shapefile
    if verbose:
        print(f"📦 正在读取基础 shapefile: {buffer_shp_path}")
    gdf = read_shapefile_with_fid(buffer_shp_path, verbose=verbose)
    
    # 2. 读取统计数据 CSV
    if verbose:
        print(f"\n📊 正在读取统计数据: {stats_csv_path}")
    stats_df = pd.read_csv(stats_csv_path)
    if verbose:
        print(f"   统计数据列: {stats_df.columns.tolist()}")
        print(f"   总记录数: {len(stats_df)}")
    
    # 3. 筛选指定 start_frame 的数据
    if verbose:
        print(f"\n🔍 筛选 start_frame == {start_frame} 的数据...")
    filtered_stats = stats_df[stats_df['start_frame'] == start_frame].copy()
    if verbose:
        print(f"   筛选后记录数: {len(filtered_stats)}")
    
    if len(filtered_stats) == 0:
        print(f"⚠️ 警告: 没有找到 start_frame == {start_frame} 的数据")
        available_frames = sorted(stats_df['start_frame'].unique())
        print(f"   可用的 start_frame 值: {available_frames[:20]}...")
        return None
    
    # 4. 将 lane_id 转换为字符串，以便与 FID 匹配
    filtered_stats['lane_id'] = filtered_stats['lane_id'].astype(str)
    
    # 5. 确保 gdf 的 fid 列是字符串类型
    gdf['fid'] = gdf['fid'].astype(str)
    
    if verbose:
        print(f"\n🔗 正在合并数据...")
        print(f"   Shapefile FID 值: {sorted(gdf['fid'].unique())[:10]}...")
        print(f"   统计数据 lane_id 值: {sorted(filtered_stats['lane_id'].unique())[:10]}...")
    
    # 6. 合并数据：将统计数据按 lane_id 与 FID 对应合并
    # 使用左连接，保留所有 shapefile 中的要素
    merged_gdf = gdf.merge(
        filtered_stats[['lane_id', 'avg_speed', 'avg_occupancy', 'total_vehicles']],
        left_on='fid',
        right_on='lane_id',
        how='left'
    )
    
    # 删除重复的 lane_id 列（如果存在）
    if 'lane_id' in merged_gdf.columns:
        merged_gdf = merged_gdf.drop(columns=['lane_id'])
    
    if verbose:
        matched_count = merged_gdf['avg_speed'].notna().sum()
        print(f"   成功匹配的要素数: {matched_count}/{len(merged_gdf)}")
    
    # 7. 添加 start_frame 列作为参考
    merged_gdf['start_frm'] = start_frame  # shapefile 字段名限制为 10 字符
    
    # 8. 保存输出 shapefile
    if verbose:
        print(f"\n💾 正在保存输出 shapefile: {output_shp_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_shp_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理空值：将数值字段中的 NaN 转换为 -1
    numeric_cols = ['avg_speed', 'avg_occupancy', 'total_vehicles']
    for col in numeric_cols:
        if col in merged_gdf.columns:
            merged_gdf[col] = merged_gdf[col].fillna(-1)
    
    merged_gdf.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')
    
    if verbose:
        print(f"✅ 完成! 输出文件: {output_shp_path}")
        print(f"   输出列: {merged_gdf.columns.tolist()}")
    
    return merged_gdf


def main():
    """主函数"""
    # 获取项目根目录（脚本所在目录的上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 配置路径（相对于项目根目录）
    buffer_shp_path = os.path.join(project_root, "plots/buffer/buffer_small_crossing_3_area.shp")
    stats_csv_path = os.path.join(project_root, "data/lane_node_stats/d210291000_lane_node_stats.csv")
    output_shp_path = os.path.join(project_root, "plots/inference/d210291000_buffer_50.shp")
    
    # 指定要筛选的 start_frame 值
    # 可以修改这个值来筛选不同时间段的数据
    target_start_frame = 50
    
    # 执行合并
    result = create_buffer_with_stats(
        buffer_shp_path=buffer_shp_path,
        stats_csv_path=stats_csv_path,
        output_shp_path=output_shp_path,
        start_frame=target_start_frame,
        verbose=True
    )
    
    if result is not None:
        print(f"\n📋 输出数据预览:")
        print(result[['fid', 'avg_speed', 'avg_occupancy', 'total_vehicles', 'start_frm']].head(10))


if __name__ == "__main__":
    main()
