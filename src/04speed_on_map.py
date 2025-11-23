# This code calculates the average speed of vehicles within each buffer segment
# for the time interval 20-30 seconds and adds the result to the shapefile

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from shapefile_utils import read_shapefile_with_fallback

# 读取分块面数据（shp文件）
print("正在读取分块面数据...")
buffer_shp_path = '../plots/buffer/d210240830.shp'

# 使用兼容性函数读取shp文件（避免fiona版本兼容性问题）
buffer_gdf = read_shapefile_with_fallback(buffer_shp_path, verbose=True)

print(f"共读取 {len(buffer_gdf)} 个分块")

# 读取车辆轨迹数据
print("正在读取车辆轨迹数据...")
df = pd.read_csv('../data/ok_data/d210240830.csv')

# 处理frame字段
df['frame'] = df['frame'].str.rstrip(';')
df['frame'] = df['frame'].astype(float)

# 筛选20~30秒区间的数据
print("正在筛选20~30秒区间的数据...")
df_filtered = df[(df['frame'] >= 20) & (df['frame'] <= 30)]
print(f"筛选后共有 {len(df_filtered)} 条记录")

# 创建车辆点的GeoDataFrame
print("正在创建车辆点几何...")
geometry = [Point(xy) for xy in zip(df_filtered.lon, df_filtered.lat)]
vehicles_gdf = gpd.GeoDataFrame(
    df_filtered[['id', 'frame', 'v']], 
    geometry=geometry, 
    crs="EPSG:4326"
)

# 确保两个GeoDataFrame使用相同的坐标系
if buffer_gdf.crs != vehicles_gdf.crs:
    vehicles_gdf = vehicles_gdf.to_crs(buffer_gdf.crs)

# 使用空间连接找出每个分块内的车辆点
print("正在进行空间连接...")
# 先为buffer_gdf添加一个唯一索引列，用于后续合并
buffer_gdf['buffer_idx'] = buffer_gdf.index
joined = gpd.sjoin(vehicles_gdf, buffer_gdf, how='inner', predicate='within')
print(f"空间连接后共有 {len(joined)} 条匹配记录")

# 按分块（使用buffer_idx）分组计算平均速度和车辆数量
print("正在计算每个分块的车辆数量...")
stats_by_buffer = joined.groupby('buffer_idx').agg({
    'v': 'mean',  # 平均速度
    'id': 'count'  # 车辆记录数量
}).reset_index()
stats_by_buffer.columns = ['buffer_idx', 'demo_20_30', 'vehicle_count']

# 将结果合并到buffer_gdf
result_gdf = buffer_gdf.merge(stats_by_buffer, on='buffer_idx', how='left')

# 对于没有车辆的分块，平均速度设为-1，车辆数量设为0
result_gdf['demo_20_30'] = result_gdf['demo_20_30'].fillna(-1)
result_gdf['vehicle_count'] = result_gdf['vehicle_count'].fillna(0).astype(int)

# 删除临时添加的buffer_idx列
result_gdf = result_gdf.drop(columns=['buffer_idx'])

# 保存更新后的shp文件
print("正在保存更新后的shp文件...")
output_path = '../plots/buffer/d210240830_2.shp'
result_gdf.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
print(f"完成！已将车辆数量添加到 {len(result_gdf)} 个分块中")
print(f"有车辆的分块数量: {len(stats_by_buffer)}")
