"""
16allNum_draw.py

将节点车辆统计数据映射到 shapefile

输入:
    - plots/buffer/d2trajectory_10_Buf.shp: 基础 shapefile
    - data/lane_node_stats/xxx_lane_node_stats.csv: 节点车辆统计数据（来自 09lane_node_NumStatic.py）

输出:
    - plots/inference/vehicle_count/xxx_vehicle_count.shp: 包含车辆数量的 shapefile
"""

import os
import pandas as pd
import geopandas as gpd
from shapefile_utils import read_shapefile_with_fallback


def main():
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 输入文件路径
    base_shp = os.path.join(project_root, "plots/buffer/d2trajectory_10_Buf.shp")
    csv_file = os.path.join(project_root, "data/lane_node_stats/d210291000_lane_node_stats.csv")
    
    # 输出目录和文件
    output_dir = os.path.join(project_root, "plots/inference/vehicle_count")
    os.makedirs(output_dir, exist_ok=True)
    
    # 从输入文件名提取输出文件名
    csv_basename = os.path.basename(csv_file).replace('_lane_node_stats.csv', '')
    output_path = os.path.join(output_dir, f"{csv_basename}_vehicle_count.shp")
    
    # 读取基础 shapefile
    print("📦 正在读取基础 Shapefile...")
    gdf_base = read_shapefile_with_fallback(base_shp, verbose=True)
    print(f"✅ 共加载 {len(gdf_base)} 个要素")
    print(f"📋 Shapefile 列名: {list(gdf_base.columns)}")
    
    # 确保 FID_ 字段存在
    if 'FID_' not in gdf_base.columns:
        print(f"❌ 错误: Shapefile 中未找到 'FID_' 字段")
        print(f"   可用字段: {list(gdf_base.columns)}")
        return
    
    # 读取 CSV 数据
    print("\n📊 正在读取 CSV 数据...")
    df = pd.read_csv(csv_file)
    print(f"✅ 共读取 {len(df)} 行数据")
    print(f"� CSV 列名:e {list(df.columns)}")
    
    # 检查必要字段
    if 'node_id' not in df.columns or 'total_vehicles' not in df.columns:
        print(f"❌ 错误: CSV 文件缺少必要字段 'node_id' 或 'total_vehicles'")
        return
    
    # 复制基础 GeoDataFrame
    gdf_result = gdf_base.copy()
    
    # 初始化 total_vehicles 字段为 0
    gdf_result['total_veh'] = 0  # 使用缩写以符合 shapefile 字段名长度限制
    
    # 遍历 CSV 中的每一行，根据 node_id 匹配 FID_
    matched_count = 0
    unmatched_nodes = []
    
    print("\n� 正在映射添数据...")
    for idx, row in df.iterrows():
        node_id = int(row['node_id'])
        total_vehicles = int(row['total_vehicles'])
        
        # 在 GeoDataFrame 中查找匹配的 FID_
        mask = gdf_result['FID_'] == node_id
        
        if mask.any():
            gdf_result.loc[mask, 'total_veh'] = total_vehicles
            matched_count += 1
        else:
            unmatched_nodes.append(node_id)
    
    print(f"✅ 成功匹配 {matched_count}/{len(df)} 个节点")
    if unmatched_nodes:
        print(f"⚠️ 未匹配的 node_id 数量: {len(unmatched_nodes)}")
        print(f"   示例: {unmatched_nodes[:10]}{'...' if len(unmatched_nodes) > 10 else ''}")
    
    # 打印统计信息
    print(f"\n📊 车辆数量统计:")
    print(f"   总节点数: {len(gdf_result)}")
    print(f"   有车辆的节点数: {(gdf_result['total_veh'] > 0).sum()}")
    print(f"   最大车辆数: {gdf_result['total_veh'].max()}")
    print(f"   最小车辆数: {gdf_result['total_veh'].min()}")
    print(f"   平均车辆数: {gdf_result['total_veh'].mean():.2f}")
    
    # 保存为新的 shapefile
    print(f"\n💾 正在保存到: {output_path}")
    gdf_result.to_file(output_path, driver='ESRI Shapefile')
    print(f"✅ 成功保存 shapefile")
    
    print(f"\n{'='*60}")
    print("🎉 Shapefile 创建完成！")
    print(f"{'='*60}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
