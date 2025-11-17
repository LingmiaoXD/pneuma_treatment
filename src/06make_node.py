# -*- coding: utf-8 -*-
"""
build_road_graph.py

基于分段车道面要素和车辆轨迹数据，
构建道路图结构，节点为 lane_segment，边分为三种类型：
- direct: 同一道路线上的前后连接
- near: 相邻车道（结合轨迹变道验证）
- crossing: 轨迹出现跳跃且距离 > 3m，且非 direct/near

输出格式：
{
  "nodes": [
    {
      "lane_id": "1",
      "node_connections": {
        "direct": [2],
        "near": [3],
        "crossing": [5]
      }
    },
    ...
  ]
}
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from collections import defaultdict


def main(lane_shp_path, traj_csv_path, output_json_path, crs="EPSG:32634"):
    """
    主函数

    参数:
        lane_shp_path: str, 车道段面要素 Shapefile 路径
        traj_csv_path: str, 轨迹 CSV 路径，含 id, frame, lon, lat 等字段
        output_json_path: str, 输出 JSON 文件路径
        crs: str, 投影坐标系（用于距离计算），希腊地区默认 UTM Zone 34N
    """
    print("🚀 开始构建道路图结构...")

    # =================== Step 1: 加载并预处理车道数据 ===================
    print("📦 正在加载车道数据...")
    # 尝试使用不同的方式读取shp文件以解决fiona版本兼容性问题
    try:
        # 方法1: 使用绝对路径和显式driver
        abs_path = os.path.abspath(lane_shp_path)
        lanes_gdf = gpd.read_file(abs_path, driver='ESRI Shapefile')
    except Exception as e1:
        # 方法2: 使用fiona直接读取（避免geopandas内部的fiona.path调用）
        try:
            import fiona
            # 直接使用fiona的open函数，避免通过geopandas
            with fiona.Env():
                with fiona.open(lane_shp_path, 'r') as src:
                    # 读取所有要素和属性
                    features = []
                    for idx, feature in enumerate(src):
                        # 确保 FID 被包含在属性中
                        if 'fid' not in feature['properties'] and 'FID' not in feature['properties']:
                            feature['properties']['fid'] = feature.get('id', idx)
                        features.append(feature)
                    # 转换为GeoDataFrame
                    lanes_gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        except Exception as e2:
            # 方法3: 使用osgeo.ogr作为备选
            try:
                from osgeo import ogr
                from shapely.geometry import shape
                import json
                
                driver = ogr.GetDriverByName('ESRI Shapefile')
                datasource = driver.Open(lane_shp_path, 0)
                layer = datasource.GetLayer()
                
                features = []
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    # 转换为shapely几何
                    geom_json = json.loads(geom.ExportToJson())
                    shapely_geom = shape(geom_json)
                    
                    # 获取属性
                    props = {}
                    for i in range(feature.GetFieldCount()):
                        field_name = feature.GetFieldDefnRef(i).GetName()
                        props[field_name] = feature.GetField(i)
                    
                    # 确保 FID 被包含（ogr 的 FID 通过 GetFID() 获取）
                    if 'fid' not in props and 'FID' not in props:
                        props['fid'] = feature.GetFID()
                    
                    features.append({
                        'geometry': shapely_geom,
                        'properties': props
                    })
                
                # 创建GeoDataFrame
                lanes_gdf = gpd.GeoDataFrame.from_features(features)
                # 尝试获取CRS
                spatial_ref = layer.GetSpatialRef()
                if spatial_ref:
                    try:
                        lanes_gdf.crs = spatial_ref.ExportToWkt()
                    except:
                        lanes_gdf.crs = "EPSG:4326"  # 默认CRS
                else:
                    lanes_gdf.crs = "EPSG:4326"
                    
            except Exception as e3:
                print(f"所有读取方法都失败了。")
                print(f"方法1错误: {e1}")
                print(f"方法2错误: {e2}")
                print(f"方法3错误: {e3}")
                print("\n建议：请更新fiona和geopandas库版本")
                print("命令: pip install --upgrade fiona geopandas")
                raise

    # 确保使用投影坐标系以正确计算距离
    if lanes_gdf.crs is None or lanes_gdf.crs.is_geographic:
        print(f"⚠️ 原始数据为地理坐标系，正在重投影到 {crs} ...")
        lanes_gdf = lanes_gdf.to_crs(crs)

    # 检查并处理 FID 字段
    print(f"📋 数据列名: {list(lanes_gdf.columns)}")
    
    # 尝试找到 FID 字段（可能是 fid, FID, 或其他变体）
    fid_col = None
    for col in lanes_gdf.columns:
        if col.lower() == 'fid':
            fid_col = col
            break
    
    if fid_col is None:
        # 如果没有找到 FID 字段，使用索引作为 FID
        print("⚠️ 未找到 FID 字段，使用索引作为 FID")
        lanes_gdf['fid'] = lanes_gdf.index.astype(str)
    else:
        # 如果找到了，使用该字段
        print(f"✅ 找到 FID 字段: {fid_col}")
        if fid_col != 'fid':
            lanes_gdf['fid'] = lanes_gdf[fid_col].astype(str)
        else:
            lanes_gdf['fid'] = lanes_gdf['fid'].astype(str)
    
    # 设置 FID 为索引
    lanes_gdf.set_index('fid', inplace=True)

    # 检查并处理 join_fid 字段
    join_fid_col = None
    for col in lanes_gdf.columns:
        if col.lower() in ['join_fid', 'JOIN_FID']:
            join_fid_col = col
            break
    
    if join_fid_col is None:
        # 如果没有找到 join_fid 字段，使用 fid 作为 join_fid（每个车道段独立）
        print("⚠️ 未找到 join_fid 字段，使用 fid 作为 join_fid（每个车道段独立）")
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
    id_to_idx = {lid: i for i, lid in idx_to_id.items()}

    NEAR_THRESHOLD = 6.0  # 米，适合城市道路宽度

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

    # ------------------- 使用轨迹验证 near -------------------
    print("🔍 正在使用轨迹数据验证 near 连接...")
    traj_df = pd.read_csv(traj_csv_path)
    traj_df = traj_df.sort_values(["id", "frame"])

    # 如果轨迹中没有 lane_id_hint，先匹配最近车道
    if 'lane_id_hint' not in traj_df.columns:
        print("📎 轨迹未标注 lane_id，正在匹配最近车道...")
        def snap_to_lane(row):
            pt = Point(row['lon'], row['lat'])
            # 转换为投影坐标系以计算距离
            pt_gdf = gpd.GeoDataFrame([1], geometry=[pt], crs="EPSG:4326")
            pt_gdf = pt_gdf.to_crs(crs)
            dists = lanes_gdf.distance(pt_gdf.geometry.iloc[0])
            return dists.idxmin()
        traj_df['lane_id_hint'] = traj_df.apply(snap_to_lane, axis=1)

    def extract_lane_changes(group):
        changes = []
        prev = None
        for _, row in group.iterrows():
            curr = str(row["lane_id_hint"])
            if prev and prev != curr:
                changes.append((prev, curr))
            prev = curr
        return changes

    change_pairs = traj_df.groupby("id").apply(extract_lane_changes).sum()
    valid_near_pairs = set(change_pairs)  # 所有真实发生过的变道

    validated_near = defaultdict(list)
    for lid in lanes_gdf.index:
        candidates = near_connections[lid]
        for nb in candidates:
            if (str(lid), str(nb)) in valid_near_pairs:
                validated_near[lid].append(nb)

    near_connections = validated_near
    print("✅ near 连接验证完成")

    # =================== Step 4: 构建 crossing 连接 ===================
    print("🚦 正在构建 crossing（交叉口）连接...")

    # 确保 lane_id_hint 已存在（在 Step 3 中可能已创建）
    if 'lane_id_hint' not in traj_df.columns:
        print("📎 轨迹未标注 lane_id，正在匹配最近车道...")
        def snap_to_lane(row):
            pt = Point(row['lon'], row['lat'])
            # 转换为投影坐标系以计算距离
            pt_gdf = gpd.GeoDataFrame([1], geometry=[pt], crs="EPSG:4326")
            pt_gdf = pt_gdf.to_crs(crs)
            dists = lanes_gdf.distance(pt_gdf.geometry.iloc[0])
            return dists.idxmin()
        traj_df['lane_id_hint'] = traj_df.apply(snap_to_lane, axis=1)

    traj_df['lane_id_hint'] = traj_df['lane_id_hint'].astype(str)
    traj_df = traj_df.sort_values(["id", "frame"])

    # 提取所有连续 lane 变化
    transitions = []
    for vid, group in traj_df.groupby("id"):
        prev_lane = None
        for _, row in group.iterrows():
            curr_lane = str(row["lane_id_hint"])
            if prev_lane and prev_lane != curr_lane:
                transitions.append((prev_lane, curr_lane))
            prev_lane = curr_lane

    unique_transitions = set(transitions)
    crossing_connections = defaultdict(list)
    CROSSING_MIN_DIST = 2.0

    def get_distance(lid1, lid2):
        try:
            p1 = lanes_gdf.loc[lid1].center_point
            p2 = lanes_gdf.loc[lid2].center_point
            return p1.distance(p2)
        except KeyError:
            return float('inf')

    for (frm, to) in unique_transitions:
        if frm == to:
            continue
        if to in direct_connections.get(frm, []) or to in near_connections.get(frm, []):
            continue
        dist = get_distance(frm, to)
        if dist < CROSSING_MIN_DIST:
            continue
        crossing_connections[frm].append(to)

    print("✅ crossing 连接构建完成")

    # =================== Step 5: 输出图结构 ===================
    print("💾 正在生成图结构 JSON...")
    graph_data = {"nodes": []}

    for lid in lanes_gdf.index:
        lid_str = str(lid)
        connections = {}

        directs = [int(x) for x in direct_connections[lid_str]]
        nears = [int(x) for x in near_connections[lid_str]]
        crossings = [int(x) for x in crossing_connections[lid_str]]

        if directs:
            connections["direct"] = directs
        if nears:
            connections["near"] = nears
        if crossings:
            connections["crossing"] = crossings

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
    TRAJ_CSV_PATH = r"../data/ok_data/d210240830.csv"         # 轨迹数据，含 id,frame,lon,lat 等字段
    OUTPUT_JSON = r"../plots/small_crossing_d210240830_graph.json"          # 输出路径

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # 执行构建
    main(LANE_SHP_PATH, TRAJ_CSV_PATH, OUTPUT_JSON)