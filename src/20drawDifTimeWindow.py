'''
    输入：
        一个节点在各时间步真值：data\draw\raw_data\three_lines_huadongtimewindow_select.csv
        要绘制的曲线是：avg_speed_1、avg_speed_2、avg_speed_3，三条线都是透明度设为50%，分别为紫色、深蓝色、绿色

    处理目标：
        start_frame列可以调整具体要显示的范围
        绘制符合期刊论文要求的图表

    输出：
        一张300dpi图表png
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_different_time_windows(df, output_path, start_frame=5, end_frame=824, dpi=300):
    """绘制不同时间窗口下的速度变化曲线
    
    参数:
        df: 包含三个时间窗口速度数据的DataFrame
        output_path: 输出图片路径
        start_frame: 起始帧
        end_frame: 结束帧
        dpi: 图片分辨率
    """
    
    # 设置图表样式 - 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建正方形图表
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 限制时间范围并按时间排序
    df = df[(df['start_frame'] >= start_frame) & (df['start_frame'] <= end_frame)].copy()
    df = df.sort_values('start_frame')
    
    # 提取时间和速度数据
    times = df['start_frame'].values
    
    # 定义三条曲线的配置
    curves = [
        {'column': 'avg_speed_1', 'color': 'purple', 'label': '时间窗口1', 'linestyle': '--'},
        {'column': 'avg_speed_2', 'color': 'darkblue', 'label': '时间窗口2', 'linestyle': '-'},
        {'column': 'avg_speed_3', 'color': 'green', 'label': '时间窗口3', 'linestyle': '--'}
    ]
    
    # 绘制三条曲线
    for curve in curves:
        speeds = df[curve['column']].values
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(speeds)
        valid_times = times[valid_mask]
        valid_speeds = speeds[valid_mask]
        
        if len(valid_times) > 0:
            ax.plot(valid_times, valid_speeds, 
                   color=curve['color'], 
                   linestyle=curve['linestyle'], 
                   linewidth=1, 
                   alpha=0.5,  # 透明度50%
                   label=curve['label'])
            
            print(f"{curve['label']}: {len(valid_times)} 个有效数据点")
            print(f"  速度范围: {valid_speeds.min():.2f} - {valid_speeds.max():.2f}")
        else:
            print(f"警告：{curve['label']} 没有有效数据")
    
    # 设置图表属性
    ax.set_xlabel('相对时间(s)', fontsize=12)
    ax.set_ylabel('平均速度', fontsize=12)
    ax.set_title(f'不同时间窗口下的速度变化曲线 (节点{df["node_id"].iloc[0] if len(df) > 0 else "N/A"})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(start_frame, end_frame)
    
    # 设置坐标轴刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    plt.close()

def main():
    # ========== 配置参数 ==========
    # 节点ID（修改此处以分析不同节点）
    NODE_ID = 1
    
    # 文件路径
    input_path = '../data/draw/raw_data/three_lines_huadongtimewindow_select.csv'
    output_path = f'../plots/node{NODE_ID}_different_time_windows.png'
    
    # 时间范围
    start_frame = 300
    end_frame = 450
    
    # 1. 加载数据
    print("加载数据...")
    df = pd.read_csv(input_path)
    
    print(f"总记录数: {len(df)} 条")
    print(f"数据列: {df.columns.tolist()}")
    
    # 2. 筛选指定node_id的数据
    print(f"\n筛选node_id={NODE_ID}的数据...")
    node_data = df[df['node_id'] == NODE_ID].copy()
    
    print(f"找到 {len(node_data)} 条记录")
    
    if len(node_data) == 0:
        print(f"错误：没有找到node_id={NODE_ID}的数据")
        return
    
    # 打印数据统计信息
    print("\n数据统计:")
    for col in ['avg_speed_1', 'avg_speed_2', 'avg_speed_3']:
        valid_count = node_data[col].notna().sum()
        if valid_count > 0:
            print(f"{col}: {valid_count} 个有效值, "
                  f"范围 {node_data[col].min():.2f} - {node_data[col].max():.2f}")
        else:
            print(f"{col}: 无有效数据")
    
    # 3. 绘制图表
    print("\n绘制图表...")
    plot_different_time_windows(node_data, output_path, start_frame, end_frame)
    
    print("\n完成!")

if __name__ == '__main__':
    main()
