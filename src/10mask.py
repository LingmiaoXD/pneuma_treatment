import pandas as pd
import ast
from shapefile_utils import read_shapefile_with_fallback

def generate_lane_mask_csv(shp_path, mask_csv_path, output_path):
    """
    生成路段掩码CSV文件
    
    Parameters:
    - shp_path: shapefile路径，包含lane_id和mask字段
    - mask_csv_path: 掩膜CSV路径，包含start_frame和mask列表
    - output_path: 输出CSV路径
    """
    # 读取shapefile，获取lane_id和mask类别
    gdf = read_shapefile_with_fallback(shp_path, verbose=True)
    lane_mask_map = dict(zip(gdf['id'], gdf['mask']))
    
    # 读取掩膜CSV
    mask_df = pd.read_csv(mask_csv_path)
    
    # 生成结果
    results = []
    for _, row in mask_df.iterrows():
        start_frame = row['start_frame']
        # 解析mask列表（可能是字符串形式）
        visible_masks = row['show']
        if isinstance(visible_masks, str):
            visible_masks = ast.literal_eval(visible_masks)
        
        # 遍历所有路段
        for lane_id, mask_val in lane_mask_map.items():
            is_observed = 1 if mask_val in visible_masks else 0
            results.append({
                'start_frame': start_frame,
                'lane_id': lane_id,
                'is_observed': is_observed
            })
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"已保存到 {output_path}")
    return result_df


if __name__ == "__main__":
    shp_path = r"../plots/buffer/buffer_small_crossing_2.shp"
    mask_csv_path = r"../data/lane_node_stats/d210291000_raw_mask.csv"
    output_path = r"../data/lane_node_stats/d210291000_lane_mask.csv"
    
    generate_lane_mask_csv(shp_path, mask_csv_path, output_path)
