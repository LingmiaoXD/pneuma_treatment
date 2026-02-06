# -*- coding: utf-8 -*-
"""
10real_node_mask.py

根据标注的可见时间段和范围，生成节点级别的mask文件

输入：
    1. 标注时间段和可见范围的csv文件：字段有start，end，duration，top_left_x，top_left_y，
       top_right_x，top_right_y，bottom_right_x，bottom_right_y，bottom_left_x，bottom_left_y
    2. shp面要素文件，里面有node_id属性字段

处理过程：
    对于输入csv里的每一行，计算这四个坐标围成的范围与shp面要素的包含关系，
    只有当一个面要素完全处于覆盖范围内时，才算作可见节点，
    将时段和可见节点的id填入输出的csv里
    例如，如果csv里一行是10.9到12.1秒，覆盖范围完全包含5个面要素，
    那么输出的csv里就对应有五行，start和end都是11到12

输出：
    一个csv文件，三个字段依次为node_id，start，end
    start和end都是由输入的csv里的信息四舍五入到接近的整数秒
"""

import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapefile_utils import read_shapefile_with_fid





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


def find_contained_nodes(polygon, nodes_gdf):
    """
    找到完全被多边形包含的所有节点面要素
    
    参数:
        polygon: shapely.geometry.Polygon, 查询多边形（覆盖范围）
        nodes_gdf: geopandas.GeoDataFrame, 节点面要素
        
    返回:
        set: 被完全包含的节点ID集合
    """
    contained_nodes = set()
    
    for idx, row in nodes_gdf.iterrows():
        # 判断面要素是否完全在覆盖范围内
        if polygon.contains(row.geometry):
            node_id = row.get('node_id')
            if pd.notna(node_id):
                contained_nodes.add(int(node_id))
    
    return contained_nodes


def round_time(time_value):
    """
    将时间四舍五入到最接近的整数秒
    
    参数:
        time_value: float, 时间值
        
    返回:
        int: 四舍五入后的整数秒
    """
    return int(round(time_value))


def main(visibility_csv_path, nodes_shp_path, output_csv_path):
    """
    主函数
    
    参数:
        visibility_csv_path: str, 标注时间段和可见范围的CSV路径
        nodes_shp_path: str, 节点面shapefile路径
        output_csv_path: str, 输出CSV路径
    """
    print("🚀 开始生成节点级别的mask文件...")
    
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
    
    # 读取节点面shapefile
    print("\n📦 正在读取节点面shapefile...")
    nodes_gdf = read_shapefile_with_fid(nodes_shp_path, verbose=True)
    
    # 确保有node_id字段
    if 'node_id' not in nodes_gdf.columns:
        raise ValueError("❌ Shapefile中缺少node_id字段")
    
    print(f"✅ 共读取 {len(nodes_gdf)} 个节点面要素")
    
    # =================== Step 2: 处理每一条标注记录 ===================
    print("\n📊 正在处理标注记录...")
    
    results = []
    
    for idx, row in visibility_df.iterrows():
        # 创建可见范围多边形
        polygon = create_polygon_from_coords(row)
        
        # 找到完全被包含的节点
        contained_nodes = find_contained_nodes(polygon, nodes_gdf)
        
        if not contained_nodes:
            print(f"⚠️ 第 {idx+1} 条记录没有找到完全包含的节点")
            continue
        
        # 四舍五入时间
        start_time = round_time(row['start'])
        end_time = round_time(row['end'])
        
        # 为每个节点生成一条记录
        for node_id in sorted(contained_nodes):
            results.append({
                'node_id': node_id,
                'start': start_time,
                'end': end_time
            })
        
        print(f"✅ 第 {idx+1} 条记录: 时间 {row['start']:.1f}-{row['end']:.1f}s "
              f"→ {start_time}-{end_time}s, "
              f"包含节点数: {len(contained_nodes)}")
    
    # =================== Step 3: 保存结果 ===================
    print(f"\n💾 正在保存结果到 {output_csv_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['node_id', 'start']).reset_index(drop=True)
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 结果已保存至: {output_csv_path}")
    print(f"📊 总计生成 {len(results_df)} 条记录")
    print(f"📊 涉及节点数: {results_df['node_id'].nunique()}")
    
    # 显示每个节点的记录数
    print("\n📊 各节点记录数统计:")
    node_counts = results_df['node_id'].value_counts().sort_index()
    for node_id, count in node_counts.items():
        print(f"  节点 {node_id}: {count} 条记录")


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # 示例路径（请根据实际情况修改）
    VISIBILITY_CSV_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\ok_data\c0127085212_0001_coverage.csv"  # 标注时间段和可见范围
    NODES_SHP_PATH = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\plots\buffer\minhang.shp"  # 节点面shapefile
    OUTPUT_CSV = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\lane_node_stats\k0127085212_0001_node_mask.csv"  # 输出路径
    
    # 检查文件是否存在
    if not os.path.exists(VISIBILITY_CSV_PATH):
        print(f"❌ 标注文件不存在: {VISIBILITY_CSV_PATH}")
        print("请修改 VISIBILITY_CSV_PATH 为实际的标注文件路径")
    elif not os.path.exists(NODES_SHP_PATH):
        print(f"❌ Shapefile不存在: {NODES_SHP_PATH}")
        print("请修改 NODES_SHP_PATH 为实际的shapefile路径")
    else:
        # 执行处理
        main(VISIBILITY_CSV_PATH, NODES_SHP_PATH, OUTPUT_CSV)