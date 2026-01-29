# 输入数据集：一个csv，包括frame_number,id,x,y,width,height,type,v(km/h)
# 输出数据集：shp格式的线要素，包括id字段

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import glob
import os

def trajectory_to_shp(csv_path, output_path):
    """
    将轨迹CSV数据转换为shapefile线要素
    
    参数:
        csv_path: 输入CSV文件路径（包含frame_number,id,x,y,width,height,type,v(km/h)）
        output_path: 输出shapefile路径
    """
    # 读取CSV数据
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 按id和frame_number排序
    df = df.sort_values(['id', 'frame_number'])
    
    # 按id分组，创建线要素
    geometries = []
    ids = []
    
    print("正在生成轨迹线要素...")
    for track_id, group in df.groupby('id'):
        # 提取坐标点（x, y）
        coords = list(zip(group['x'], group['y']))
        
        # 至少需要2个点才能构成线
        if len(coords) >= 2:
            line = LineString(coords)
            geometries.append(line)
            ids.append(track_id)
    
    # 创建GeoDataFrame
    # 使用UTM Zone 51N投影坐标系 (EPSG:32651)
    gdf = gpd.GeoDataFrame({
        'id': ids
    }, geometry=geometries, crs="EPSG:32651")
    
    # 保存为shapefile
    print(f"正在保存shapefile: {output_path}")
    gdf.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
    print(f"完成！共生成 {len(gdf)} 条轨迹线")


if __name__ == "__main__":
    # 输入数据路径
    input_dir = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\ok_data"
    output_dir = r"E:\大学文件\研二\交通分析\代码\pneuma_treatment\yolodata\derived\trajectory"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有轨迹数据文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    print(f"找到 {len(csv_files)} 个轨迹数据文件")
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        # 生成输出文件名
        basename = os.path.basename(csv_file).replace('.csv', '')
        output_file = os.path.join(output_dir, f"{basename}_trajectory.shp")
        
        # 转换
        trajectory_to_shp(csv_file, output_file)
        print("-" * 50)
