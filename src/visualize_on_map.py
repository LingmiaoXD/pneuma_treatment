# This code plots the time-series traffic state data
# with 10-second intervals and 20-meter spatial segments
# showing average speed for each segment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import utm
from sklearn.decomposition import PCA
from shapely.geometry import Point, LineString
from geopandas import GeoDataFrame

# 配置中文字体支持
# Windows系统常用字体：SimHei（黑体）、Microsoft YaHei（微软雅黑）
# matplotlib会自动选择列表中第一个可用的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

SIZE = 20
plt.rc('font', size=SIZE)
plt.rc('axes', titlesize=SIZE)
plt.rc('axes', labelsize=SIZE)
plt.rc('xtick', labelsize=SIZE)
plt.rc('ytick', labelsize=SIZE)
plt.rc('legend', fontsize=SIZE)
plt.rc('figure', titlesize=SIZE)

def determine_road_direction(df):
    """
    使用PCA确定道路的主要方向
    返回道路方向向量、中心点和UTM区域信息
    """
    # 使用所有轨迹点来确定主要方向
    coords = np.column_stack([df['lon'].values, df['lat'].values])
    
    # 转换为UTM坐标，使用第一个有效点的zone
    utm_coords = []
    zone_num = None
    zone_letter = None
    
    for lon, lat in coords:
        try:
            easting, northing, zn, zl = utm.from_latlon(lat, lon)
            if zone_num is None:
                zone_num = zn
                zone_letter = zl
            # 只使用相同zone的点
            if zn == zone_num and zl == zone_letter:
                utm_coords.append([easting, northing])
        except:
            continue
    
    if len(utm_coords) == 0:
        raise ValueError("无法转换UTM坐标")
    
    utm_coords = np.array(utm_coords)
    
    # 计算中心点
    center = utm_coords.mean(axis=0)
    
    # 使用PCA找到主要方向
    pca = PCA(n_components=2)
    pca.fit(utm_coords - center)
    
    # 主要方向是第一个主成分
    road_direction = pca.components_[0]
    
    return road_direction, center, zone_num, zone_letter

def project_to_road(utm_coords, road_direction, center):
    """
    将UTM坐标投影到道路方向上
    返回沿道路的距离（米）
    """
    # 将坐标相对于中心点
    centered_coords = utm_coords - center
    
    # 投影到道路方向
    distances = np.dot(centered_coords, road_direction)
    
    return distances

def create_spatial_segments(distances, segment_length=20):
    """
    创建空间分段，每segment_length米一个分隔
    返回每个点的分段索引
    """
    # 找到最小距离作为起点
    min_dist = distances.min()
    
    # 计算每个点属于哪个分段
    segment_indices = ((distances - min_dist) / segment_length).astype(int)
    
    return segment_indices, min_dist

# change the path to the data that you want to visualize
df = pd.read_csv('../data/ok_data/d210240830.csv')
#TODO: 考虑meta数据里的车型，把车型也在图中显示

# 处理frame字段
df['frame'] = df['frame'].str.rstrip(';')
df['frame'] = df['frame'].astype(float)

# 确定道路方向
print("正在计算道路方向...")
road_direction, center, zone_num, zone_letter = determine_road_direction(df)

# 为所有数据点计算UTM坐标和沿道路的距离
print("正在计算空间分段...")
utm_data = []
for idx, row in df.iterrows():
    try:
        easting, northing, zn, zl = utm.from_latlon(row['lat'], row['lon'])
        # 只使用相同zone的点
        if zn == zone_num and zl == zone_letter:
            utm_data.append([easting, northing])
        else:
            utm_data.append([np.nan, np.nan])
    except:
        utm_data.append([np.nan, np.nan])

utm_array = np.array(utm_data)
valid_mask = ~np.isnan(utm_array[:, 0])
df_valid = df[valid_mask].copy()
utm_valid = utm_array[valid_mask]

# 计算沿道路的距离
road_distances = project_to_road(utm_valid, road_direction, center)

# 创建空间分段（20米一个分隔）
segment_indices, min_dist = create_spatial_segments(road_distances, segment_length=20)
df_valid['road_distance'] = road_distances
df_valid['segment_index'] = segment_indices

# 获取边界用于绘图
min_x = df.lon.min()
max_x = df.lon.max()
min_y = df.lat.min()
max_y = df.lat.max()

# 时间间隔：10秒
time_interval = 10.0
frame = 0

print("正在生成可视化图像...")
while frame < df.frame.max():
    
    # 获取当前10秒时间窗口内的数据
    temp_sample = df_valid[(df_valid.frame >= frame) & (df_valid.frame < frame + time_interval)].copy()
    
    if len(temp_sample) == 0:
        frame += time_interval
        continue
    
    # 计算每个空间分段内的平均速度
    segment_stats = temp_sample.groupby('segment_index').agg({
        'v': 'mean',
        'lon': 'mean',
        'lat': 'mean',
        'road_distance': 'mean'
    }).reset_index()
    
    segment_stats.columns = ['segment_index', 'avg_speed', 'lon', 'lat', 'road_distance']
    
    # 创建图形
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 使用散点图显示每个分段，颜色表示平均速度
    scatter = ax.scatter(segment_stats['lon'], 
                        segment_stats['lat'], 
                        c=segment_stats['avg_speed'],
                        s=100,  # 较大的点表示分段
                        alpha=0.7,
                        vmin=0,
                        vmax=60,
                        cmap='RdYlGn',
                        edgecolors='black',
                        linewidths=1)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.0)
    cbar.set_label('平均速度 (km/h)', rotation=90, labelpad=20)
    
    # 设置坐标轴范围
    ax.set_ylim([min_y, max_y])
    ax.set_xlim([min_x, max_x])
    
    # 添加速度标注（可选，如果分段不太多的话）
    if len(segment_stats) <= 50:  # 只在分段不太多时显示标注
        for idx, row in segment_stats.iterrows():
            ax.annotate(f'{row["avg_speed"]:.1f}', 
                       (row['lon'], row['lat']),
                       fontsize=8, ha='center', va='center',
                       color='white' if row['avg_speed'] < 30 else 'black',
                       weight='bold')
    
    # 绘制北方向箭头
    x, y, arrow_length = 0.1, 0.95, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 设置标题，显示时间范围
    ax.set_title(f"时间: {frame:.1f}s - {frame+time_interval:.1f}s | 分段数: {len(segment_stats)}")
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = "../plots/after_visualization_d2_10240830/"
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + f"{int(frame)}.png", dpi=100)
    plt.close()
    
    # 时间间隔：10秒
    frame += time_interval
    
print("可视化完成！")
    