"""
16time_allnode_draw.py

生成时空特征图，筛选指定时间的数据，给每个节点增加相应的属性

输入:
    - plots/buffer/d2trajectory_10_Buf.shp: 基础 shapefile
    - data/draw/d210191000/d210291000_lane_node_stats.csv: 时空统计数据

输出:
    - plots/inference/time_allnode/439/true/d210291000_lane_node_stats_439.shp
"""

import os
import pandas as pd
import geopandas as gpd
from shapefile_utils import read_shapefile_with_fallback


def main():
    # ========== 配置参数 ==========
    TARGET_TIME = 142  # 要筛选的时间值
    # ==============================
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 输入文件路径
    base_shp = os.path.join(project_root, "plots/buffer/d2trajectory_10_Buf.shp")
    csv_file = os.path.join(project_root, "data/draw/d210191000/st_idw_results.csv")
    
    # 输出目录
    output_dir = os.path.join(project_root, "plots/inference/time_allnode")
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
    
    # 筛选 time 列为 TARGET_TIME 的行
    print(f"\n🔍 筛选 time={TARGET_TIME} 的行...")
    df = df[df['time'] == TARGET_TIME].copy()
    print(f"✅ 筛选后共 {len(df)} 行数据")
    
    # 处理重复的 node_id，保留第一次出现的行
    if df.duplicated(subset=['node_id']).any():
        duplicate_count = df.duplicated(subset=['node_id']).sum()
        print(f"⚠️ 发现 {duplicate_count} 个重复的 node_id，将保留第一次出现的行")
        df = df.drop_duplicates(subset=['node_id'], keep='first')
        print(f"✅ 去重后共 {len(df)} 行数据")
    
    print(f"\n{'='*60}")
    print(f"🔧 生成 shapefile")
    print(f"{'='*60}")
    
    # 复制基础 GeoDataFrame
    gdf_result = gdf_base.copy()
    
    # 获取 CSV 中的所有列（除了 node_id）
    data_columns = [col for col in df.columns if col != 'node_id']
    print(f"📋 要添加的数据列: {data_columns}")
    
    # 为每个数据列在 GeoDataFrame 中初始化为 -1
    for col in data_columns:
        # 初始化为 -1，表示空值
        gdf_result[col] = -1.0
    
    # 遍历 CSV 中的每一行，根据 node_id 匹配 FID_
    matched_count = 0
    unmatched_nodes = []
    
    for idx, row in df.iterrows():
        node_id = int(row['node_id'])
        
        # 在 GeoDataFrame 中查找匹配的 FID_
        mask = gdf_result['FID_'] == node_id
        
        if mask.any():
            # 将该行的所有数据列添加到对应的 FID_，最多保留4位小数
            for col in data_columns:
                value = row[col]
                # 如果是数值类型，最多保留4位小数（不足4位保持原样）
                if pd.notna(value) and isinstance(value, (int, float)):
                    # 先转为浮点数并四舍五入到4位小数
                    rounded_value = round(float(value), 4)
                    # 保持为浮点数类型，即使是整数值
                    gdf_result.loc[mask, col] = float(rounded_value)
                else:
                    # 如果是空值，设置为 -1
                    gdf_result.loc[mask, col] = -1.0
            matched_count += 1
        else:
            unmatched_nodes.append(node_id)
    
    print(f"✅ 成功匹配 {matched_count}/{len(df)} 个节点")
    if unmatched_nodes:
        print(f"⚠️ 未匹配的 node_id: {unmatched_nodes[:10]}{'...' if len(unmatched_nodes) > 10 else ''}")
    
    # 确保所有数据列都是浮点数类型，并将 NaN 替换为 -1
    print(f"\n🔧 正在转换数据类型...")
    for col in data_columns:
        gdf_result[col] = pd.to_numeric(gdf_result[col], errors='coerce')
        # 将 NaN 替换为 -1
        gdf_result[col] = gdf_result[col].fillna(-1.0)
    print(f"✅ 数据类型转换完成")
    
    # 打印数据类型信息
    print(f"\n📊 字段数据类型:")
    for col in data_columns:
        print(f"   {col}: {gdf_result[col].dtype}")
    
    # 保存为新的 shapefile
    # 从 CSV 文件名中提取基础名称（不含扩展名）
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    output_filename = f"{csv_basename}_{TARGET_TIME}.shp"
    output_path = os.path.join(output_dir, output_filename)
    print(f"\n💾 正在保存到: {output_path}")
    gdf_result.to_file(output_path, driver='ESRI Shapefile')
    print(f"✅ 成功保存 {output_filename}")
    
    print(f"\n{'='*60}")
    print("🎉 Shapefile 创建完成！")
    print(f"{'='*60}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
