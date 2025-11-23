# -*- coding: utf-8 -*-
"""
build_road_graph.py

基于分段车道面要素构建道路图结构，节点为 lane_segment，边分为两种类型：
- direct: 同一道路线上的前后连接
- near: 相邻车道（基于空间距离判断）

输出格式：
{
  "nodes": [
    {
      "lane_id": "1",
      "node_connections": {
        "direct": [2],
        "near": [3]
      }
    },
    ...
  ]
}
"""

import os
import json
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


def main(lane_shp_path, output_json_path, crs="EPSG:32634"):
    """
    主函数

    参数:
        lane_shp_path: str, 车道段面要素 Shapefile 路径
        output_json_path: str, 输出 JSON 文件路径
        crs: str, 投影坐标系（用于距离计算），希腊地区默认 UTM Zone 34N
    """
    print("🚀 开始构建道路图结构...")

    # =================== Step 1: 加载并预处理车道数据 ===================
    print("📦 正在加载车道数据...")
    # 读取 Shapefile
    lanes_gdf = gpd.read_file(lane_shp_path)
    
    # 确保使用投影坐标系以正确计算距离
    if lanes_gdf.crs is None or lanes_gdf.crs.is_geographic:
        print(f"⚠️ 原始数据为地理坐标系，正在重投影到 {crs} ...")
        lanes_gdf = lanes_gdf.to_crs(crs)
    
    # 设置 id 为索引
    lanes_gdf.set_index('id', inplace=True)

    # 检查并处理 join_fid 字段
    join_fid_col = None
    for col in lanes_gdf.columns:
        if col.lower() in ['join_fid', 'JOIN_FID']:
            join_fid_col = col
            break
    
    if join_fid_col is None:
        # 如果没有找到 join_fid 字段，使用 id 作为 join_fid（每个车道段独立）
        print("⚠️ 未找到 join_fid 字段，使用 id 作为 join_fid（每个车道段独立）")
        lanes_gdf['join_fid'] = lanes_gdf.index
    else:
        # 如果找到了，使用该字段
        print(f"✅ 找到 join_fid 字段: {join_fid_col}")
        if join_fid_col != 'join_fid':
            lanes_gdf['join_fid'] = lanes_gdf[join_fid_col]
        # 如果已经是 join_fid，不需要额外处理

    # 添加中心点列
    lanes_gdf['center_point'] = lanes_gdf.centroid

    # 获取所有 lane_id 列表
    all_lane_ids = list(lanes_gdf.index)
    print(f"✅ 共加载 {len(all_lane_ids)} 个车道段")

    # =================== Step 2: 构建 direct 连接 ===================
    print("🔗 正在构建 direct（前后直联）连接...")
    direct_connections = defaultdict(list)

    for road_id, group in lanes_gdf.groupby('join_fid'):
        if len(group) <= 1:
            continue

        line_geom = group.iloc[0].geometry.convex_hull.boundary  # 近似主线路线方向
        # 使用质心投影到主方向向量排序
        coords = [(row.center_point.x, row.center_point.y) for _, row in group.iterrows()]
        coords = np.array(coords)
        cx, cy = coords[:, 0], coords[:, 1]
        mean_x, mean_y = np.mean(cx), np.mean(cy)
        dx, dy = cx - mean_x, cy - mean_y
        angles = np.arctan2(dy, dx)
        sorted_indices = np.argsort(angles)

        sorted_lanes = group.iloc[sorted_indices].index.tolist()
        for i in range(len(sorted_lanes) - 1):
            curr = str(sorted_lanes[i])
            nxt = str(sorted_lanes[i+1])
            direct_connections[curr].append(nxt)
            # 反向不自动添加（单向道）

    print(f"✅ direct 连接构建完成")

    # =================== Step 3: 构建 near 连接（相邻车道）===================
    print("↔️ 正在构建 near（相邻车道）连接...")
    near_connections = defaultdict(list)

    # 提取所有中心点坐标
    coords = np.array([[pt.x, pt.y] for pt in lanes_gdf.center_point])
    tree = cKDTree(coords)
    idx_to_id = {i: lid for i, lid in enumerate(lanes_gdf.index)}

    NEAR_THRESHOLD = 3.0  # 米，适合城市道路宽度

    for i, (lid, row) in enumerate(lanes_gdf.iterrows()):
        center = row.center_point
        indices = tree.query_ball_point([center.x, center.y], r=NEAR_THRESHOLD)
        for j in indices:
            if i == j:
                continue
            neighbor_id = idx_to_id[j]
            # 排除同一路线上的（那是 direct）
            if row['join_fid'] == lanes_gdf.loc[neighbor_id]['join_fid']:
                continue
            near_connections[lid].append(neighbor_id)

    print(f"✅ near 连接构建完成，共 {sum(len(v) for v in near_connections.values())} 条连接")

    # =================== Step 4: 输出图结构 ===================
    print("💾 正在生成图结构 JSON...")
    graph_data = {"nodes": []}

    for lid in lanes_gdf.index:
        lid_str = str(lid)
        connections = {}

        directs = [int(x) for x in direct_connections[lid_str]]
        nears = [int(x) for x in near_connections[lid_str]]

        if directs:
            connections["direct"] = directs
        if nears:
            connections["near"] = nears

        graph_data["nodes"].append({
            "lane_id": lid_str,
            "node_connections": connections
        })

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 图结构已保存至: {output_json_path}")
    print(f"📊 总计节点数: {len(graph_data['nodes'])}")


# =================== 示例调用 ===================
if __name__ == "__main__":

    LANE_SHP_PATH = r"../plots/buffer/buffer_small_crossing_2.shp"        # 车道段面数据
    OUTPUT_JSON = r"../data/road_graph/small_crossing_d210240830_graph.json"                   # 输出路径

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # 执行构建
    main(LANE_SHP_PATH, OUTPUT_JSON)