# 先处理轨迹数据文件，只留下经过目标buffer的轨迹点，便于后面处理
# 增加车道段ID和car_type两列

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from shapefile_utils import read_shapefile_with_fid


if __name__ == "__main__":
    LANE_SHP_PATH = r"../plots/buffer/buffer_small_crossing_2.shp"        # 车道段面数据
    TRAJ_CSV_PATH = r"../data/ok_data/d210240830.csv"         # 轨迹数据，含 id,frame,lon,lat 等字段
    TRAJ_META_PATH = r"../data/ok_data/meta_d210240830.csv"        # 轨迹元数据，含 id,type等字段
    OUTPUT_CSV = r"../data/trajectory_with_laneid/d210240830.csv"          # 输出路径
    
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # =================== Step 1: 读取车道段面数据 ===================
    print("正在读取车道段面数据...")
    # 使用工具函数读取 Shapefile 并确保 FID 正确
    lane_gdf = read_shapefile_with_fid(LANE_SHP_PATH, crs=None, set_fid_as_index=False, verbose=True)
    
    # =================== Step 2: 读取轨迹数据 ===================
    print("正在读取轨迹数据...")
    traj_df = pd.read_csv(TRAJ_CSV_PATH)
    
    # 处理frame字段（如果有分号）
    if 'frame' in traj_df.columns:
        traj_df['frame'] = traj_df['frame'].astype(str).str.rstrip(';')
        traj_df['frame'] = traj_df['frame'].astype(float)
    
    print(f"共读取 {len(traj_df)} 条轨迹记录")
    
    # =================== Step 3: 创建轨迹点的GeoDataFrame ===================
    print("正在创建轨迹点几何...")
    geometry = [Point(xy) for xy in zip(traj_df.lon, traj_df.lat)]
    traj_gdf = gpd.GeoDataFrame(
        traj_df,
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    # 确保两个GeoDataFrame使用相同的坐标系
    if lane_gdf.crs != traj_gdf.crs:
        traj_gdf = traj_gdf.to_crs(lane_gdf.crs)
    
    # =================== Step 4: 空间连接获取车道段ID ===================
    print("正在进行空间连接...")
    # 使用空间连接找出每个点在哪个面要素内
    joined = gpd.sjoin(traj_gdf, lane_gdf, how='left', predicate='within')
    
    # 如果有多行匹配（一个点匹配多个面），只保留第一个匹配
    # 使用索引来匹配回原始的traj_df
    if len(joined) > len(traj_df):
        print(f"⚠️ 检测到多行匹配（{len(joined)} 行 vs {len(traj_df)} 行），将只保留第一个匹配")
        # 按索引去重，保留每个点的第一个匹配
        joined = joined[~joined.index.duplicated(keep='first')]
    
    # 确保索引对齐
    joined = joined.reindex(traj_df.index)
    
    # 提取FID作为车道段ID（使用之前创建的fid字段）
    if 'fid' in joined.columns:
        traj_df['FID'] = joined['fid'].values
    else:
        # 如果空间连接后没有fid字段，使用index_right
        print("⚠️ 空间连接后未找到fid字段，使用index_right作为车道段ID")
        traj_df['FID'] = joined.index_right.astype(str) if hasattr(joined.index_right, 'astype') else joined.index_right
    
    print(f"空间连接完成，共 {len(traj_df)} 条记录")
    
    # 立即过滤掉没有匹配上车道段ID的记录，避免后续冗余计算
    print("正在过滤数据，只保留有车道段ID的记录...")
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna()].copy()
    filtered_count = len(traj_df)
    print(f"过滤前: {original_count} 条记录，过滤后: {filtered_count} 条记录")
    
    # =================== Step 5: 连接元数据获取car_type ===================
    print("正在读取轨迹元数据...")
    meta_df = pd.read_csv(TRAJ_META_PATH)
    
    # 通过id字段连接获取type字段
    if 'type' in meta_df.columns:
        traj_df = traj_df.merge(
            meta_df[['id', 'type']],
            on='id',
            how='left'
        )
        traj_df['car_type'] = traj_df['type']
        traj_df = traj_df.drop(columns=['type'])
    else:
        print("警告: 元数据中未找到type字段")
        traj_df['car_type'] = None
    
    print(f"元数据连接完成")
    
    # =================== Step 6: 保存结果 ===================
    print(f"正在保存结果到 {OUTPUT_CSV}...")
    traj_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print(f"完成！共处理 {filtered_count} 条轨迹记录（仅包含有车道段ID的记录）")
    print(f"有车道段ID的记录数: {traj_df['FID'].notna().sum()}")
    print(f"有car_type的记录数: {traj_df['car_type'].notna().sum()}")