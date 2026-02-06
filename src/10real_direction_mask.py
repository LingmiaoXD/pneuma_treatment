# -*- coding: utf-8 -*-
"""
10real_node_mask.py

根据标注的可见时间段和范围，生成方向级别的mask文件

输入：
    1. 标注时间段和可见范围的csv文件：字段有start，end，duration，top_left_x，top_left_y，
       top_right_x，top_right_y，bottom_right_x，bottom_right_y，bottom_left_x，bottom_left_y
    2. shp线要素文件，里面有lane_id
    3. 图结构文件，参考minhang_graph.json

处理过程：
    对于输入csv里的每一行，计算这四个坐标围成的范围与shp要素存在相交的有哪些线，
    在图结构文件里查找这些线属于哪些方向，将时段和可见方向的id填入输出的csv里
    例如，如果csv里一行是10.9到12.1秒，覆盖的shp线要素有5条，经过查询图结构发现来自4个方向，
    那么输出的csv里就对应有四行，start和end都是11到12

输出：
    一个csv文件，三个字段依次为direction_id，start，end
    start和end都是由输入的csv里的信息四舍五入到接近的整数秒
"""

import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapefile_utils import read_shapefile_with_fid


def load_graph(graph_json_path):
    """
    加载图结构，构建 lane_id 到 direction_id 的映射
    
    参数:
        graph_json_path: str, graph.json文件路径
        
    返回:
        dict: {lane_id: [direction_ids]} 一个车道可能属于多个方向
    """
    print(f"📦 正在读取图结构: {graph_json_path}")
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 构建 lane_id -> direction_ids 的映射
    lane_to_directions = {}
    for direction in graph_data.get('directions', []):
        direction_id = direction['direction_id']
        lanes = direction.get('lanes', [])
        
        for lane_id in lanes:
            if lane_id not in lane_to_directions:
                lane_to_directions[lane_id] = []
            lane_to_directions[lane_id].append(direction_id)
    
    print(f"✅ 共加载 {len(lane_to_directions)} 个车道到方向的映射")
    print(f"✅ 共有 {len(graph_data.get('directions', []))} 个方向")
    
    return lane_to_directions


def create_polygon_from_coords(row):
    """
    从CSV行数据创建多边形
    
    参数:
        row: pandas.Series, 包含四个角点坐标的行
        
    返回:
        shapely.geometry.Polygon: 多边形对象
    """
    coords = [
        (row['top_left_x'], row['top_left_y']),
        (row['top_right_x'], row['top_right_y']),
        (row['bottom_right_x'], row['bottom_right_y']),
        (row['bottom_left_x'], row['bottom_left_y']),
        (row['top_left_x'], row['top_left_y'])  # 闭合多边形
    ]
    return Polygon(coords)


def find_intersecting_lanes(polygon, lanes_gdf):
    """
    找到与多边形相交的所有车道
    
    参数:
        polygon: shapely.geometry.Polygon, 查询多边形
        lanes_gdf: geopandas.GeoDataFrame, 车道线要素
        
    返回:
        set: 相交的车道ID集合
    """
    # 使用空间索引加速查询
    intersecting_lanes = set()
    
    for idx, row in lanes_gdf.iterrows():
        if polygon.intersects(row.geometry):
            lane_id = row.get('lane_id')
            if pd.notna(lane_id):
                intersecting_lanes.add(int(lane_id))
    
    return intersecting_lanes


def round_time(time_value):
    """
    将时间四舍五入到最接近的整数秒
    
    参数:
        time_value: float, 时间值
        
    返回:
        int: 四舍五入后的整数秒
    """
    return int(round(time_value))


def main(visibility_csv_path, lanes_shp_path, graph_json_path, output_csv_path):
    """
    主函数
    
    参数:
        visibility_csv_path: str, 标注时间段和可见范围的CSV路径
        lanes_shp_path: str, 车道线shapefile路径
        graph_json_path: str, graph.json文件路径
        output_csv_path: str, 输出CSV路径
    """
    print("🚀 开始生成方向级别的mask文件...")
    
    # =================== Step 1: 读取数据 ===================
    print("\n📦 正在读取标注数据...")
    visibility_df = pd.read_csv(visibility_csv_path)
    
    # 检查必要字段
    required_fields = ['start', 'end', 'top_left_x', 'top_left_y', 'top_right_x', 
                      'top_right_y', 'bottom_right_x', 'bottom_right_y', 
                      'bottom_left_x', 'bottom_left_y']
    missing_fields = [f for f in required_fields if f not in visibility_df.columns]
    if missing_fields:
        raise ValueError(f"❌ 标注数据缺少必要字段: {missing_fields}")
    
    print(f"✅ 共读取 {len(visibility_df)} 条标注记录")
    
    # 读取车道线shapefile
    print("\n📦 正在读取车道线shapefile...")
    lanes_gdf = read_shapefile_with_fid(lanes_shp_path, verbose=True)
    
    # 确保有lane_id字段
    if 'lane_id' not in lanes_gdf.columns:
        raise ValueError("❌ Shapefile中缺少lane_id字段")
    
    print(f"✅ 共读取 {len(lanes_gdf)} 条车道线要素")
    
    # 加载图结构
    lane_to_directions = load_graph(graph_json_path)
    
    # =================== Step 2: 处理每一条标注记录 ===================
    print("\n📊 正在处理标注记录...")
    
    results = []
    
    for idx, row in visibility_df.iterrows():
        # 创建可见范围多边形
        polygon = create_polygon_from_coords(row)
        
        # 找到相交的车道
        intersecting_lanes = find_intersecting_lanes(polygon, lanes_gdf)
        
        if not intersecting_lanes:
            print(f"⚠️ 第 {idx+1} 条记录没有找到相交的车道")
            continue
        
        # 找到这些车道对应的方向
        directions_set = set()
        for lane_id in intersecting_lanes:
            if lane_id in lane_to_directions:
                directions_set.update(lane_to_directions[lane_id])
        
        if not directions_set:
            print(f"⚠️ 第 {idx+1} 条记录的车道没有对应的方向")
            continue
        
        # 四舍五入时间
        start_time = round_time(row['start'])
        end_time = round_time(row['end'])
        
        # 为每个方向生成一条记录
        for direction_id in sorted(directions_set):
            results.append({
                'direction_id': direction_id,
                'start': start_time,
                'end': end_time
            })
        
        print(f"✅ 第 {idx+1} 条记录: 时间 {row['start']:.1f}-{row['end']:.1f}s "
              f"→ {start_time}-{end_time}s, "
              f"车道数: {len(intersecting_lanes)}, 方向数: {len(directions_set)}")
    
    # =================== Step 3: 保存结果 ===================
    print(f"\n💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['direction_id', 'start']).reset_index(drop=True)
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 结果已保存至: {output_csv_path}")
    print(f"📊 总计生成 {len(results_df)} 条记录")
    print(f"📊 涉及方向数: {results_df['direction_id'].nunique()}")
    
    # 显示每个方向的记录数
    print("\n📊 各方向记录数统计:")
    direction_counts = results_df['direction_id'].value_counts().sort_index()
    for direction_id, count in direction_counts.items():
        print(f"  {direction_id}: {count} 条记录")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # 示例路径（请根据实际情况修改）
    VISIBILITY_CSV_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\ok_data\c0127085212_0001_coverage.csv"  # 标注时间段和可见范围
    LANES_SHP_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\plots\buffer\minhang_raw_line.shp"  # 车道线shapefile
    GRAPH_JSON_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\data\road_graph\minhang_graph.json"  # 图结构
    OUTPUT_CSV = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\minhang_lane_node_stats\k0127085212_0001_patrol_mask.csv"  # 输出路径
    
    # 检查文件是否存在
    if not os.path.exists(VISIBILITY_CSV_PATH):
        print(f"❌ 标注文件不存在: {VISIBILITY_CSV_PATH}")
        print("请修改 VISIBILITY_CSV_PATH 为实际的标注文件路径")
    elif not os.path.exists(LANES_SHP_PATH):
        print(f"❌ Shapefile不存在: {LANES_SHP_PATH}")
        print("请修改 LANES_SHP_PATH 为实际的shapefile路径")
    elif not os.path.exists(GRAPH_JSON_PATH):
        print(f"❌ 图文件不存在: {GRAPH_JSON_PATH}")
        print("请修改 GRAPH_JSON_PATH 为实际的图文件路径")
    else:
        # 执行处理
        main(VISIBILITY_CSV_PATH, LANES_SHP_PATH, GRAPH_JSON_PATH, OUTPUT_CSV)