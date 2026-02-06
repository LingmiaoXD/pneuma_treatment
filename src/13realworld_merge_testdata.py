'''
13realworld_merge_testdata.py

输入：
    两个csv文件A和B，列名均为node_id，start_frame，avg_speed，avg_occupancy，total_vehicles

处理：
    B文件相当于真值
    对于B文件里total_vehicles不为空的行，将这一行的avg_speed，avg_occupancy，total_vehicles替代A行相同node_id和start_frame的这几个属性

输出：
    修改后的A文件
'''

import os
import pandas as pd
import numpy as np


def merge_test_data(file_a_path, file_b_path, output_path):
    """
    将B文件（真值）中的数据合并到A文件中
    
    参数:
        file_a_path: str, A文件路径（待更新的文件）
        file_b_path: str, B文件路径（真值文件）
        output_path: str, 输出文件路径
    """
    print("🚀 开始合并测试数据...")
    
    # =================== Step 1: 读取数据 ===================
    print("\n📦 正在读取文件A（待更新文件）...")
    df_a = pd.read_csv(file_a_path)
    print(f"✅ 文件A共有 {len(df_a)} 条记录")
    
    print("\n📦 正在读取文件B（真值文件）...")
    df_b = pd.read_csv(file_b_path)
    print(f"✅ 文件B共有 {len(df_b)} 条记录")
    
    # 检查必要字段
    required_fields = ['node_id', 'start_frame', 'avg_speed', 'avg_occupancy', 'total_vehicles']
    
    missing_fields_a = [f for f in required_fields if f not in df_a.columns]
    if missing_fields_a:
        raise ValueError(f"❌ 文件A缺少必要字段: {missing_fields_a}")
    
    missing_fields_b = [f for f in required_fields if f not in df_b.columns]
    if missing_fields_b:
        raise ValueError(f"❌ 文件B缺少必要字段: {missing_fields_b}")
    
    # =================== Step 2: 过滤B文件中有效的真值数据 ===================
    print("\n🔍 正在过滤B文件中的有效真值数据...")
    
    # 只保留total_vehicles不为空的行
    df_b_valid = df_b[df_b['total_vehicles'].notna()].copy()
    print(f"✅ B文件中有效真值记录数: {len(df_b_valid)}")
    
    if len(df_b_valid) == 0:
        print("⚠️ B文件中没有有效的真值数据（total_vehicles全为空），直接保存A文件")
        df_a.to_csv(output_path, index=False, encoding='utf-8')
        print(f"💾 结果已保存至: {output_path}")
        return df_a
    
    # =================== Step 3: 合并数据 ===================
    print("\n🔄 正在合并数据...")
    
    # 确保数据类型一致
    df_a['node_id'] = df_a['node_id'].astype(str)
    df_a['start_frame'] = df_a['start_frame'].astype(int)
    
    df_b_valid['node_id'] = df_b_valid['node_id'].astype(str)
    df_b_valid['start_frame'] = df_b_valid['start_frame'].astype(int)
    
    # 创建合并键
    df_a['merge_key'] = df_a['node_id'] + '_' + df_a['start_frame'].astype(str)
    df_b_valid['merge_key'] = df_b_valid['node_id'] + '_' + df_b_valid['start_frame'].astype(str)
    
    # 统计匹配情况
    matched_keys = set(df_a['merge_key']) & set(df_b_valid['merge_key'])
    print(f"📊 B文件与A文件匹配的记录数: {len(matched_keys)}")
    
    # 使用B文件的值更新A文件
    update_count = 0
    skip_count = 0  # A文件total_vehicles为空，跳过的记录数
    
    for idx, row_a in df_a.iterrows():
        merge_key = row_a['merge_key']
        
        # 查找B文件中对应的行
        matching_rows = df_b_valid[df_b_valid['merge_key'] == merge_key]
        
        if not matching_rows.empty:
            # 检查A文件当前行的total_vehicles是否为空
            if pd.isna(row_a['total_vehicles']):
                # A文件的total_vehicles为空，跳过不替换
                skip_count += 1
                continue
            
            # A文件和B文件的total_vehicles都不为空，执行替换
            # 取第一条匹配记录（理论上应该只有一条）
            row_b = matching_rows.iloc[0]
            
            # 更新avg_speed, avg_occupancy, total_vehicles
            df_a.at[idx, 'avg_speed'] = row_b['avg_speed']
            df_a.at[idx, 'avg_occupancy'] = row_b['avg_occupancy']
            df_a.at[idx, 'total_vehicles'] = row_b['total_vehicles']
            
            update_count += 1
    
    print(f"✅ 成功更新 {update_count} 条记录")
    print(f"⚠️ 跳过 {skip_count} 条记录（A文件total_vehicles为空）")
    
    # 删除辅助列
    df_a = df_a.drop(columns=['merge_key'])
    
    # =================== Step 4: 保存结果 ===================
    print(f"\n💾 正在保存结果到 {output_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存CSV文件
    df_a.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"🎉 合并结果已保存至: {output_path}")
    print(f"📊 输出记录数: {len(df_a)}")
    print(f"📊 更新记录数: {update_count}")
    print(f"📊 跳过记录数: {skip_count}")
    print(f"📊 更新比例: {update_count / len(df_a):.2%}")
    
    return df_a


# =================== 示例调用 ===================
if __name__ == "__main__":
    
    # =================== 配置参数 ===================
    FILE_A_PATH = r"../yolodata/minhang_lane_node_stats/k0127085212_0001_test_2.csv"  # 待更新的文件
    FILE_B_PATH = r"../yolodata/minhang_lane_node_stats/k0127085203_0001_lane_node_state.csv"  # 真值文件
    OUTPUT_PATH = r"../yolodata/minhang_lane_node_stats/k0127085212_0001_test.csv"  # 输出路径
    
    # 检查文件是否存在
    if not os.path.exists(FILE_A_PATH):
        raise FileNotFoundError(f"❌ 文件A不存在: {FILE_A_PATH}")
    
    if not os.path.exists(FILE_B_PATH):
        raise FileNotFoundError(f"❌ 文件B不存在: {FILE_B_PATH}")
    
    # 执行合并
    merge_test_data(FILE_A_PATH, FILE_B_PATH, OUTPUT_PATH)