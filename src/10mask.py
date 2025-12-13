import pandas as pd
import ast
import os
from shapefile_utils import read_shapefile_with_fallback

def generate_raw_mask_csv(output_path, start_frame_range=(0, 830), show_values=[0, 1, 2, 3], start_show_index=2, duration=1):
    """
    生成raw_mask CSV文件
    
    Parameters:
    - output_path: 输出CSV路径
    - start_frame_range: start_frame的范围，元组(start, end)，包含end
    - show_values: show的所有可能值列表，例如[0, 1, 2, 3]
    - start_show_index: 起始show值在show_values中的索引，例如2表示从[2]开始
    - duration: 每个show值持续的frame数，默认为1
    """
    start_frame, end_frame = start_frame_range
    results = []
    
    # 计算起始show值
    current_show_index = start_show_index % len(show_values)
    frame_count = 0
    
    for frame in range(start_frame, end_frame + 1):
        # 获取当前show值
        current_show = show_values[current_show_index]
        results.append({
            'start_frame': frame,
            'show': f'[{current_show}]'
        })
        
        frame_count += 1
        # 如果达到duration，切换到下一个show值
        if frame_count >= duration:
            frame_count = 0
            current_show_index = (current_show_index + 1) % len(show_values)
    
    # 保存结果
    result_df = pd.DataFrame(results)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_df.to_csv(output_path, index=False)
    print(f"已生成 {len(result_df)} 条记录")
    print(f"已保存到 {output_path}")
    print(f"\n前10条记录预览:")
    print(result_df.head(10))
    print(f"\n后10条记录预览:")
    print(result_df.tail(10))
    
    return result_df

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

def generate_lane_mask_from_params(shp_path, output_path, 
                                    mask_csv_path=None,
                                    start_frame_range=(0, 830), 
                                    show_values=[0, 1, 2, 3], 
                                    start_show_index=2, 
                                    duration=1):
    """
    从参数直接生成lane_mask CSV文件（整合raw_mask生成和lane_mask生成）
    
    Parameters:
    - shp_path: shapefile路径，包含lane_id和mask字段
    - output_path: 最终输出的lane_mask CSV路径
    - mask_csv_path: 已有的raw_mask CSV路径，如果提供且文件存在则直接使用，跳过生成步骤；如果为None或文件不存在则生成并保存
    - start_frame_range: start_frame的范围，元组(start, end)，包含end（仅在需要生成raw_mask时使用）
    - show_values: show的所有可能值列表，例如[0, 1, 2, 3]（仅在需要生成raw_mask时使用）
    - start_show_index: 起始show值在show_values中的索引，例如2表示从[2]开始（仅在需要生成raw_mask时使用）
    - duration: 每个show值持续的frame数，默认为1（仅在需要生成raw_mask时使用）
    """
    # 判断是否需要生成raw_mask文件
    if mask_csv_path is not None and os.path.exists(mask_csv_path):
        # 如果提供了路径且文件存在，直接使用
        print("=" * 50)
        print(f"检测到已有的raw_mask文件: {mask_csv_path}")
        print("跳过生成步骤，直接使用现有文件")
        print("=" * 50)
    else:
        # 如果没有提供路径或文件不存在，生成raw_mask CSV路径
        if mask_csv_path is None:
            output_dir = os.path.dirname(output_path)
            output_basename = os.path.basename(output_path)
            mask_csv_path = os.path.join(output_dir, output_basename.replace('_lane_mask.csv', '_raw_mask.csv'))
        
        # 第一步：生成raw_mask CSV
        print("=" * 50)
        print("步骤1: 生成raw_mask CSV")
        print("=" * 50)
        generate_raw_mask_csv(
            output_path=mask_csv_path,
            start_frame_range=start_frame_range,
            show_values=show_values,
            start_show_index=start_show_index,
            duration=duration
        )
    
    # 第二步：从raw_mask生成lane_mask
    print("\n" + "=" * 50)
    print("步骤2: 从raw_mask生成lane_mask CSV")
    print("=" * 50)
    generate_lane_mask_csv(shp_path, mask_csv_path, output_path)
    
    return output_path


if __name__ == "__main__":
    shp_path = r"../plots/buffer/buffer_small_crossing_4.shp"
    output_path = r"../data/lane_node_stats/d210291000_lane_mask.csv"
    
    # 方式1: 如果没有raw_mask文件，自动生成并保存
    generate_lane_mask_from_params(
        shp_path=shp_path,
        output_path=output_path,
        mask_csv_path=None,  # 不提供路径，自动生成
        start_frame_range=(0, 830),
        show_values=[0, 1, 2, 3],
        start_show_index=2,  # 从[2]开始
        duration=5  # 每个值持续5个frame
    )
    
    # 方式2: 如果已有raw_mask文件，直接提供路径，跳过生成步骤
    # mask_csv_path = r"../data/lane_node_stats/d210291000_raw_mask.csv"
    # generate_lane_mask_from_params(
    #     shp_path=shp_path,
    #     output_path=output_path,
    #     mask_csv_path=mask_csv_path  # 提供已有文件路径，跳过生成
    # )
