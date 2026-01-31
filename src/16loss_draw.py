"""
16loss_draw.py

生成时空误差热力图，遍历 CSV 中所有值，给每个节点增加相应的属性

输入:
    - plots/buffer/d2trajectory_10_Buf.shp: 基础 shapefile
    - data/model_output/all_metrics.csv: 时空误差统计数据

输出:
    - plots/inference/loss_map/avg_speed.shp
    - plots/inference/loss_map/avg_occupancy.shp
    - plots/inference/loss_map/total_vehicles.shp
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
    csv_file = os.path.join(project_root, "data/model_output/all_metrics.csv")
    
    # 输出目录
    output_dir = os.path.join(project_root, "plots/inference/loss_map3")
    os.makedirs(output_dir, exist_ok=True)
    
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
    print(f"📋 CSV 列名: {list(df.columns)}")
    
    # 获取所有唯一的 metric 类型
    metrics = df['metric'].unique()
    print(f"\n📌 发现的 metric 类型: {metrics}")
    
    # 为每个 metric 创建一个 shapefile
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"� 处理 emetric: {metric}")
        print(f"{'='*60}")
        
        # 复制基础 GeoDataFrame
        gdf_metric = gdf_base.copy()
        
        # 筛选当前 metric 的数据
        df_metric = df[df['metric'] == metric].copy()
        print(f"📊 该 metric 共有 {len(df_metric)} 行数据")
        
        # 获取 CSV 中的所有列（除了 node_id 和 metric）
        data_columns = [col for col in df_metric.columns if col not in ['node_id', 'metric']]
        print(f"📋 要添加的数据列: {data_columns}")
        
        # 为每个数据列在 GeoDataFrame 中初始化为浮点数类型
        for col in data_columns:
            # 初始化为 NaN（浮点数类型），而不是 None
            gdf_metric[col] = float('nan')
        
        # 遍历 CSV 中的每一行，根据 node_id 匹配 FID_
        matched_count = 0
        unmatched_nodes = []
        
        for idx, row in df_metric.iterrows():
            node_id = int(row['node_id'])
            
            # 在 GeoDataFrame 中查找匹配的 FID_
            mask = gdf_metric['FID_'] == node_id
            
            if mask.any():
                # 将该行的所有数据列添加到对应的 FID_，最多保留4位小数
                for col in data_columns:
                    value = row[col]
                    # 如果是数值类型，最多保留4位小数（不足4位保持原样）
                    if pd.notna(value) and isinstance(value, (int, float)):
                        # 先转为浮点数并四舍五入到4位小数
                        rounded_value = round(float(value), 4)
                        # 保持为浮点数类型，即使是整数值
                        gdf_metric.loc[mask, col] = float(rounded_value)
                    else:
                        # 如果不是数值，保持为 NaN
                        gdf_metric.loc[mask, col] = float('nan')
                matched_count += 1
            else:
                unmatched_nodes.append(node_id)
        
        print(f"✅ 成功匹配 {matched_count}/{len(df_metric)} 个节点")
        if unmatched_nodes:
            print(f"⚠️ 未匹配的 node_id: {unmatched_nodes[:10]}{'...' if len(unmatched_nodes) > 10 else ''}")
        
        # 确保所有数据列都是浮点数类型
        print(f"\n🔧 正在转换数据类型...")
        for col in data_columns:
            gdf_metric[col] = pd.to_numeric(gdf_metric[col], errors='coerce')
        print(f"✅ 数据类型转换完成")
        
        # 打印数据类型信息
        print(f"\n📊 字段数据类型:")
        for col in data_columns:
            print(f"   {col}: {gdf_metric[col].dtype}")
        
        # 保存为新的 shapefile
        output_path = os.path.join(output_dir, f"{metric}.shp")
        print(f"\n💾 正在保存到: {output_path}")
        gdf_metric.to_file(output_path, driver='ESRI Shapefile')
        print(f"✅ 成功保存 {metric}.shp")
    
    print(f"\n{'='*60}")
    print("🎉 所有 shapefile 创建完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    for metric in metrics:
        print(f"  - {metric}.shp")


if __name__ == "__main__":
    main()

