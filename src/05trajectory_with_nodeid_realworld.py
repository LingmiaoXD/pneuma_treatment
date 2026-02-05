# 先处理轨迹数据文件，只留下经过目标buffer的轨迹点，便于后面处理
# 增加FID、lane_id和car_type三列

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
from shapefile_utils import read_shapefile_with_fallback


if __name__ == "__main__":
    LANE_SHP_PATH = r"../plots/buffer/minhang.shp"        # 车道段面数据
    TRAJ_CSV_PATH = r"../yolodata/ok_data/0127085203_0001.csv"         # 轨迹数据，含 track_id,frame_number,corrected_x,corrected_y,width,height,class_name,speed_kmh 等字段
    OUTPUT_CSV = r"../yolodata/trajectory_with_laneid/0127085203_0001.csv"          # 输出路径
    
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
    if 'frame_number' in traj_df.columns:
        traj_df['frame_number'] = traj_df['frame_number'].astype(str).str.rstrip(';')
        traj_df['frame_number'] = traj_df['frame_number'].astype(float)
    
    # 重命名字段以保持与原输出格式一致
    # track_id -> id, frame_number -> frame
    if 'track_id' in traj_df.columns:
        traj_df.rename(columns={'track_id': 'id'}, inplace=True)
    if 'frame_number' in traj_df.columns:
        traj_df.rename(columns={'frame_number': 'frame'}, inplace=True)
    
    print(f"共读取 {len(traj_df)} 条轨迹记录")
    print(f"📋 traj_df 的所有列名: {list(traj_df.columns)}")
    
    # =================== Step 3: 创建轨迹点的GeoDataFrame ===================
    print("正在创建轨迹点几何...")
    # 使用 corrected_x, corrected_y（投影坐标）创建几何
    geometry = [Point(xy) for xy in zip(traj_df.corrected_x, traj_df.corrected_y)]
    traj_gdf = gpd.GeoDataFrame(
        traj_df,
        geometry=geometry,
        crs=lane_gdf.crs  # 使用与车道数据相同的坐标系
    )
    
    # 坐标系已经在创建时设置为一致，无需转换
    
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
    
    # 调试：打印 lane_gdf 中 node_id 和 lane_id 的值范围
    print(f"📊 lane_gdf 字段值范围调试:")
    if 'node_id' in lane_gdf.columns:
        print(f"   - node_id 范围: {lane_gdf['node_id'].min()} ~ {lane_gdf['node_id'].max()}")
    if 'lane_id' in lane_gdf.columns:
        print(f"   - lane_id 范围: {lane_gdf['lane_id'].min()} ~ {lane_gdf['lane_id'].max()}")
    
    # 提取节点ID（使用 node_id 作为连接标识）
    if 'node_id' in joined.columns:
        print(f"✅ 使用 'node_id' 字段作为节点ID")
        # 使用 .values 确保索引对齐
        traj_df['FID'] = joined['node_id'].values
        print(f"   - 连接后 FID 范围: {traj_df['FID'].min()} ~ {traj_df['FID'].max()}")
        print(f"   - 连接后 FID 唯一值数量: {traj_df['FID'].nunique()}")
    else:
        print("❌ 错误：空间连接后未找到 node_id 字段")
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
    
    # =================== Step 5: 处理车辆类型 ===================
    print("正在处理车辆类型...")
    # class_name 字段已经在轨迹数据中，直接映射为 car_type
    if 'class_name' in traj_df.columns:
        # 类型映射：car -> car, bus -> medium, truck -> heavy, motorcycle -> motorcycle
        type_mapping = {
            'car': 'car',
            'bus': 'medium',
            'truck': 'heavy',
            'motorcycle': 'motorcycle',
            'van': 'medium'
        }
        # 转换为小写后映射
        traj_df['car_type'] = traj_df['class_name'].str.lower().map(type_mapping)
        
        # 过滤掉没有映射的记录
        before_filter = len(traj_df)
        traj_df = traj_df[traj_df['car_type'].notna()].copy()
        after_filter = len(traj_df)
        print(f"   - 过滤前: {before_filter} 条记录")
        print(f"   - 过滤后: {after_filter} 条记录")
        print(f"   - 删除了 {before_filter - after_filter} 条未映射的记录")
    else:
        print("警告: 轨迹数据中未找到class_name字段")
        traj_df['car_type'] = None
    
    print(f"车辆类型处理完成")
    
    
    # =================== Step 6: 保存结果 ===================
    print(f"正在保存结果到 {OUTPUT_CSV}...")
    traj_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print(f"完成！共处理 {filtered_count} 条轨迹记录（仅包含有车道段ID的记录）")
    print(f"有车道段ID的记录数: {traj_df['FID'].notna().sum()}")
    print(f"有car_type的记录数: {traj_df['car_type'].notna().sum()}")