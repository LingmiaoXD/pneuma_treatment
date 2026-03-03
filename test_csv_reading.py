"""
测试CSV文件读取和数据清洗
"""
import pandas as pd

file_path = 'data/draw/d210191000/d210291000_lane_node_stats.csv'

print("测试CSV文件读取...")
print(f"文件路径: {file_path}")

try:
    # 尝试读取CSV文件
    df = pd.read_csv(
        file_path,
        on_bad_lines='skip',
        engine='python',
        encoding='utf-8',
        skipinitialspace=True
    )
    
    print(f"\n成功读取文件！")
    print(f"数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())
    
    # 检查node_id=42的数据
    node_42_data = df[df['node_id'] == 42]
    print(f"\n节点42的数据行数: {len(node_42_data)}")
    
    if len(node_42_data) > 0:
        # 清洗速度数据
        df['avg_speed'] = pd.to_numeric(df['avg_speed'], errors='coerce')
        node_42_data = df[df['node_id'] == 42]
        speeds = node_42_data['avg_speed'].dropna()
        
        print(f"节点42有效速度数据点数: {len(speeds)}")
        print(f"速度范围: {speeds.min():.2f} - {speeds.max():.2f}")
        print(f"平均速度: {speeds.mean():.2f}")
        
        # 限制时间范围
        node_42_range = node_42_data[(node_42_data['time'] >= 34) & (node_42_data['time'] <= 824)]
        speeds_range = node_42_range['avg_speed'].dropna()
        print(f"\n时间范围[34-824]内的速度数据点数: {len(speeds_range)}")
        
        if len(speeds_range) >= 3:
            print("✓ 数据足够进行聚类分析")
        else:
            print("✗ 数据不足，无法进行聚类分析")
    else:
        print("✗ 未找到节点42的数据")
        
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
