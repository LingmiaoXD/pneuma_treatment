# 先处理轨迹数据文件，只留下经过目标buffer的轨迹点，便于后面处理
# 增加FID、lane_id和car_type三列

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
from shapefile_utils import read_shapefile_with_fallback


if __name__ == "__main__":
    LANE_SHP_PATH = r"../plots/buffer_10/d2trajectory_10_Buffer.shp"        # 车道段面数据
    TRAJ_CSV_PATH = r"../data/ok_data/d210240900.csv"         # 轨迹数据，含 id,frame,lon,lat 等字段
    TRAJ_META_PATH = r"../data/ok_data/meta_d210240900.csv"        # 轨迹元数据，含 id,type等字段
    OUTPUT_CSV = r"../data/trajectory_with_laneid/d210240900.csv"          # 输出路径
    
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # =================== Step 1: 读取车道段面数据 ===================
    print("正在读取车道段面数据...")
    # 读取 Shapefile（使用兼容性函数避免版本问题）
    lane_gdf = read_shapefile_with_fallback(LANE_SHP_PATH, verbose=True)
    
    # 打印所有属性字段名称，用于调试
    print(f"📋 lane_gdf 的所有属性字段名称: {list(lane_gdf.columns)}")
    print(f"📋 lane_gdf 的索引类型: {type(lane_gdf.index).__name__}")
    
    # =================== Step 2: 读取轨迹数据 ===================
    print("正在读取轨迹数据...")
    traj_df = pd.read_csv(TRAJ_CSV_PATH)
    
    # 处理frame字段（如果有分号）
    if 'frame' in traj_df.columns:
        traj_df['frame'] = traj_df['frame'].astype(str).str.rstrip(';')
        traj_df['frame'] = traj_df['frame'].astype(float)
    
    print(f"共读取 {len(traj_df)} 条轨迹记录")
    print(f"📋 traj_df 的所有列名: {list(traj_df.columns)}")
    
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

    print(f"📊 空间连接匹配情况:")
    print(f"   - 总轨迹点数: {len(traj_gdf)}")
    print(f"   - 匹配上的点数: {joined['index_right'].notna().sum()}")
    print(f"   - lane_gdf CRS: {lane_gdf.crs}")
    print(f"   - traj_gdf CRS: {traj_gdf.crs}")
    
    # 如果有多行匹配（一个点匹配多个面），只保留第一个匹配
    # 使用索引来匹配回原始的traj_df
    if len(joined) > len(traj_df):
        print(f"⚠️ 检测到多行匹配（{len(joined)} 行 vs {len(traj_df)} 行），将只保留第一个匹配")
        # 按索引去重，保留每个点的第一个匹配
        joined = joined[~joined.index.duplicated(keep='first')]
    
    # 确保索引对齐
    joined = joined.reindex(traj_df.index)
    
    # 打印空间连接后的所有列名，用于调试
    print(f"📋 空间连接后 joined 的所有列名: {list(joined.columns)}")
    
    # 调试：打印 lane_gdf 中 FID_ 和 fid 的值范围
    print(f"📊 lane_gdf 字段值范围调试:")
    if 'FID_' in lane_gdf.columns:
        print(f"   - FID_ 范围: {lane_gdf['FID_'].min()} ~ {lane_gdf['FID_'].max()}")
    if 'fid' in lane_gdf.columns:
        print(f"   - fid 范围: {lane_gdf['fid'].min()} ~ {lane_gdf['fid'].max()}")
    if 'lane_id' in lane_gdf.columns:
        print(f"   - lane_id 范围: {lane_gdf['lane_id'].min()} ~ {lane_gdf['lane_id'].max()}")
    
    # 提取车道段ID（使用 FID_ 或 fid 作为连接标识）
    fid_field = None
    for field in ['FID_', 'fid']:
        if field in joined.columns:
            fid_field = field
            break
    
    if fid_field:
        print(f"✅ 使用 '{fid_field}' 字段作为车道段FID")
        # 使用 .values 确保索引对齐
        traj_df['FID'] = joined[fid_field].values
        print(f"   - 连接后 FID 范围: {traj_df['FID'].min()} ~ {traj_df['FID'].max()}")
        print(f"   - 连接后 FID 唯一值数量: {traj_df['FID'].nunique()}")
    else:
        print("❌ 错误：空间连接后未找到 FID_ 或 fid 字段")
        print(f"   可用的列名: {list(joined.columns)}")
        print(f"   lane_gdf 的列名: {list(lane_gdf.columns)}")
        sys.exit(1)
    
    # 提取 lane_id（车道ID，用于输出）
    if 'lane_id' in joined.columns:
        print(f"✅ 找到 lane_id 字段，添加到输出")
        traj_df['lane_id'] = joined['lane_id'].values
    else:
        print("⚠️ 未找到 lane_id 字段")
    
    print(f"空间连接完成，共 {len(traj_df)} 条记录")
    
    # 立即过滤掉没有匹配上车道段ID的记录，避免后续冗余计算
    print("正在过滤数据，只保留有车道段ID的记录...")
    original_count = len(traj_df)
    traj_df = traj_df[traj_df['FID'].notna() & (traj_df['FID'] != 'nan')].copy()
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
        # 类型映射：Car/Taxi -> car, Bus/Medium Vehicle -> medium, Heavy Vehicle -> heavy, Motorcycle -> motorcycle
        type_mapping = {
            'Car': 'car',
            'Taxi': 'car',
            'Bus': 'medium',
            'Medium Vehicle': 'medium',
            'Heavy Vehicle': 'heavy',
            'Motorcycle': 'motorcycle'
        }
        traj_df['car_type'] = traj_df['type'].map(type_mapping)
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