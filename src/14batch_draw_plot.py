# -*- coding: utf-8 -*-
"""
14batch_plot.py

批量生成分级色彩地图，遍历 CSV 中所有 start_frame 值，
为每个 start_frame 生成对应的 PNG 格式地图。

连续渐变色彩规则（可自定义字段）:
    - 灰色: < 0 或无数据
    - 连续渐变: 5=红色, 15=橙色, 30=黄色, 60=绿色, 120=深绿色
    - 中间值为连续过渡颜色

输入:
    - plots/buffer/buffer_small_crossing_3_area.shp: 基础 shapefile
    - data/lane_node_stats/d210291000_lane_node_stats.csv: 车道统计数据

输出:
    - plots/inference/maps/: 批量生成的 PNG 地图
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colorbar as mcolorbar
import numpy as np
from shapefile_utils import read_shapefile_with_fid


# 定义连续渐变色的关键点: 5=红色, 15=橙色, 30=黄色, 60=绿色, 120=深绿色
COLOR_ANCHORS = [
    (5, '#FF0000'),    # 红色
    (15, '#FFA500'),   # 橙色
    (30, '#FFFF00'),   # 黄色
    (60, '#00FF00'),   # 绿色
    (120, '#006400'),  # 深绿色
]

# 创建连续渐变 colormap
def create_continuous_colormap():
    """创建从红到深绿的连续渐变 colormap"""
    values = [anchor[0] for anchor in COLOR_ANCHORS]
    colors = [anchor[1] for anchor in COLOR_ANCHORS]
    
    # 归一化到 0-1 范围
    min_val, max_val = values[0], values[-1]
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # 创建 colormap
    cmap = LinearSegmentedColormap.from_list('speed_cmap', list(zip(normalized, colors)))
    return cmap, min_val, max_val


def get_continuous_color(value, cmap, vmin, vmax):
    """
    根据值返回连续渐变颜色
    
    规则:
        - 灰色: < 0 或无数据
        - 连续渐变: 5=红色, 15=橙色, 30=黄色, 60=绿色, 120=深绿色
    """
    if pd.isna(value) or value < 0:
        return '#808080'  # 灰色
    
    # 将值限制在范围内
    clamped = max(vmin, min(vmax, value))
    # 归一化
    normalized = (clamped - vmin) / (vmax - vmin)
    # 获取颜色
    rgba = cmap(normalized)
    # 转换为 hex
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


def create_map_for_frame(
    gdf: gpd.GeoDataFrame,
    stats_df: pd.DataFrame,
    start_frame: float,
    output_path: str,
    color_field: str = 'avg_speed',
    title_prefix: str = '',
    figsize: tuple = (12, 10),
    dpi: int = 150
):
    """
    为指定的 start_frame 生成分级色彩地图
    
    参数:
        gdf: GeoDataFrame, 基础 shapefile 数据
        stats_df: DataFrame, 统计数据
        start_frame: float, 要筛选的 start_frame 值
        output_path: str, 输出 PNG 路径
        color_field: str, 用于分级色彩的字段名
        title_prefix: str, 标题前缀
        figsize: tuple, 图片尺寸
        dpi: int, 图片分辨率
    
    返回:
        bool: 是否成功生成
    """
    # 筛选指定 start_frame 的数据
    filtered_stats = stats_df[stats_df['time'] == start_frame].copy()
    
    if len(filtered_stats) == 0:
        return False
    
    # 准备合并
    filtered_stats['node_id'] = filtered_stats['node_id'].astype(str)
    gdf_copy = gdf.copy()
    gdf_copy['fid'] = gdf_copy['fid'].astype(str)
    
    # 合并数据
    merged_gdf = gdf_copy.merge(
        filtered_stats[['node_id', color_field]],
        left_on='fid',
        right_on='node_id',
        how='left'
    )
    
    # 创建连续渐变 colormap
    cmap, vmin, vmax = create_continuous_colormap()
    
    # 计算每个要素的颜色（连续渐变）
    merged_gdf['color'] = merged_gdf[color_field].apply(
        lambda v: get_continuous_color(v, cmap, vmin, vmax)
    )
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 绘制地图
    merged_gdf.plot(
        ax=ax,
        color=merged_gdf['color'],
        edgecolor='black',
        linewidth=0.3
    )
    
    # 创建连续渐变色条 (colorbar) 在右下角
    # 添加一个小的 axes 用于 colorbar
    cbar_ax = fig.add_axes([0.72, 0.15, 0.02, 0.25])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(color_field, fontsize=10)
    cbar.set_ticks([5, 15, 30, 60, 120])
    
    # 添加灰色图例（无数据）在 colorbar 下方
    gray_patch = mpatches.Patch(facecolor='#808080', edgecolor='black', label='nodata (< 0)')
    ax.legend(handles=[gray_patch], loc='lower right', fontsize=9)
    
    # 设置标题
    title = f'{title_prefix}Frame {int(start_frame)} - {color_field}'
    ax.set_title(title, fontsize=14)
    
    # 移除坐标轴
    ax.set_axis_off()
    
    # 保存图片 (不使用 tight_layout，因为手动添加的 colorbar axes 不兼容)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return True


def batch_generate_maps(
    buffer_shp_path: str,
    stats_csv_path: str,
    output_dir: str,
    color_field: str = 'avg_speed',
    start_frame_begin: int = None,
    start_frame_end: int = None,
    frame_interval: int = 1,
    title_prefix: str = '',
    figsize: tuple = (12, 10),
    dpi: int = 150,
    verbose: bool = True
):
    """
    批量生成分级色彩地图
    
    参数:
        buffer_shp_path: str, 基础 buffer shapefile 路径
        stats_csv_path: str, 车道统计数据 CSV 路径
        output_dir: str, 输出目录
        color_field: str, 用于分级色彩的字段名 (如 'avg_speed', 'avg_occupancy', 'total_vehicles')
        start_frame_begin: int, 开始 frame (None 表示从最小值开始)
        start_frame_end: int, 结束 frame (None 表示到最大值结束)
        frame_interval: int, frame 间隔 (默认 1，即连续)
        title_prefix: str, 标题前缀
        figsize: tuple, 图片尺寸
        dpi: int, 图片分辨率
        verbose: bool, 是否打印详细信息
    
    返回:
        int: 成功生成的地图数量
    """
    # 1. 读取基础 shapefile
    if verbose:
        print(f"📦 正在读取基础 shapefile: {buffer_shp_path}")
    gdf = read_shapefile_with_fid(buffer_shp_path, verbose=False)
    
    # 2. 读取统计数据 CSV
    if verbose:
        print(f"📊 正在读取统计数据: {stats_csv_path}")
    stats_df = pd.read_csv(stats_csv_path)
    
    # 检查 color_field 是否存在
    if color_field not in stats_df.columns:
        print(f"❌ 错误: 字段 '{color_field}' 不存在于 CSV 中")
        print(f"   可用字段: {stats_df.columns.tolist()}")
        return 0
    
    # 3. 获取所有可用的 start_frame 值
    all_frames = sorted(stats_df['time'].unique())
    
    if verbose:
        print(f"   总 frame 数: {len(all_frames)}")
        print(f"   frame 范围: {all_frames[0]} - {all_frames[-1]}")
    
    # 4. 根据参数筛选 frame 范围
    if start_frame_begin is None:
        start_frame_begin = int(all_frames[0])
    if start_frame_end is None:
        start_frame_end = int(all_frames[-1])
    
    # 生成目标 frame 列表
    target_frames = [f for f in all_frames 
                     if start_frame_begin <= f <= start_frame_end 
                     and (f - start_frame_begin) % frame_interval == 0]
    
    if verbose:
        print(f"\n🎯 目标 frame 范围: {start_frame_begin} - {start_frame_end}, 间隔: {frame_interval}")
        print(f"   将生成 {len(target_frames)} 张地图")
    
    # 5. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"📁 创建输出目录: {output_dir}")
    
    # 6. 批量生成地图
    success_count = 0
    total = len(target_frames)
    
    for i, frame in enumerate(target_frames):
        output_path = os.path.join(output_dir, f"frame_{int(frame):05d}.png")
        
        success = create_map_for_frame(
            gdf=gdf,
            stats_df=stats_df,
            start_frame=frame,
            output_path=output_path,
            color_field=color_field,
            title_prefix=title_prefix,
            figsize=figsize,
            dpi=dpi
        )
        
        if success:
            success_count += 1
            if verbose and (i + 1) % 10 == 0:
                print(f"   进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
    
    if verbose:
        print(f"\n✅ 完成! 成功生成 {success_count}/{total} 张地图")
        print(f"   输出目录: {output_dir}")
    
    return success_count


def main():
    """主函数"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # ========== 配置参数 ==========
    
    # 输入文件路径
    buffer_shp_path = os.path.join(project_root, "plots/buffer/d2trajectory_10_Buf.shp")
    stats_csv_path = os.path.join(project_root, "data/lane_node_stats/d210291000_lane_node_stats.csv")
    
    # 输出目录
    output_dir = os.path.join(project_root, "plots/test")
    
    # 分级色彩字段 (可选: 'avg_speed', 'avg_occupancy', 'total_vehicles')
    color_field = 'avg_speed'
    
    # Frame 筛选参数 (设为 None 表示使用全部范围)
    start_frame_begin = None  # 开始 frame
    start_frame_end = None    # 结束 frame
    frame_interval = 1        # 间隔 (1 表示连续)
    
    # 图片参数
    title_prefix = 'D2 10:29-10:00 '  # 标题前缀
    figsize = (12, 10)                # 图片尺寸
    dpi = 150                         # 分辨率
    
    # ========== 执行批量生成 ==========
    
    batch_generate_maps(
        buffer_shp_path=buffer_shp_path,
        stats_csv_path=stats_csv_path,
        output_dir=output_dir,
        color_field=color_field,
        start_frame_begin=start_frame_begin,
        start_frame_end=start_frame_end,
        frame_interval=frame_interval,
        title_prefix=title_prefix,
        figsize=figsize,
        dpi=dpi,
        verbose=True
    )


if __name__ == "__main__":
    main()
