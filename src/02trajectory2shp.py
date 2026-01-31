# 输入数据集：csv格式，包括id,frame,lon,lat,v,a_x,a_y
# 输出数据集：shp格式的线要素，包括id字段

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import glob
import os

def trajectory_to_shp(csv_path, output_path, meta_path=None):
    """
    将轨迹CSV数据转换为shapefile线要素
    
    参数:
        csv_path: 输入CSV文件路径
        output_path: 输出shapefile路径
        meta_path: 元数据文件路径（可选），用于过滤Motorcycle
    """
    # 读取CSV数据
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 读取metadata并过滤Motorcycle
    if meta_path and os.path.exists(meta_path):
        print(f"正在读取轨迹元数据: {meta_path}")
        meta_df = pd.read_csv(meta_path)
        
        if 'type' in meta_df.columns:
            # 合并轨迹数据和元数据
            df = df.merge(
                meta_df[['id', 'type']],
                on='id',
                how='left'
            )
            # 过滤掉Motorcycle类型
            original_count = len(df)
            df = df[df['type'] != 'Motorcycle'].copy()
            filtered_count = len(df)
            print(f"过滤前: {original_count} 条记录，过滤后: {filtered_count} 条记录（已移除 Motorcycle）")
            # 删除type列
            df = df.drop(columns=['type'])
        else:
            print("警告: 元数据中未找到type字段，未进行过滤")
    else:
        if meta_path:
            print(f"警告: 元数据文件不存在: {meta_path}，未进行过滤")
    
    # 处理frame字段（参考visualize_on_map.py）
    df['frame'] = df['frame'].str.rstrip(';')
    df['frame'] = df['frame'].astype(float)
    
    # 按id和frame排序
    df = df.sort_values(['id', 'frame'])
    
    # 按id分组，创建线要素
    geometries = []
    ids = []
    
    print("正在生成轨迹线要素...")
    for track_id, group in df.groupby('id'):
        # 提取坐标点（lon, lat）
        coords = list(zip(group['lon'], group['lat']))
        
        # 至少需要2个点才能构成线
        if len(coords) >= 2:
            line = LineString(coords)
            geometries.append(line)
            ids.append(track_id)
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'id': ids
    }, geometry=geometries, crs="EPSG:4326")
    
    # 保存为shapefile
    print(f"正在保存shapefile: {output_path}")
    gdf.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
    print(f"完成！共生成 {len(gdf)} 条轨迹线")


if __name__ == "__main__":
    # 输入数据路径
    input_dir = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\data\ok_data_d6"
    output_dir = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\data\derived\trajectory2"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有轨迹数据文件（排除meta文件）
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('meta_')]
    
    print(f"找到 {len(csv_files)} 个轨迹数据文件")
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        # 生成输出文件名
        basename = os.path.basename(csv_file).replace('.csv', '')
        output_file = os.path.join(output_dir, f"{basename}_trajectory.shp")
        
        # 查找对应的meta文件
        meta_file = os.path.join(input_dir, f"meta_{basename}.csv")
        
        # 转换
        trajectory_to_shp(csv_file, output_file, meta_path=meta_file)
        print("-" * 50)
