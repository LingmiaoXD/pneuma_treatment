'''
    输入：
        多个CSV文件，每个文件包含节点的时间序列数据，比较不同轮巡周期的情况下

    处理目标：
        绘制多条曲线的对比图表，支持用户自定义颜色和样式

    思路：
        从多个CSV文件中读取指定node_id的数据
        在同一张图上绘制多条曲线，每条曲线的颜色、线型、标签由用户定义

    输出：
        一张高分辨率的对比图表
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_curves(data_configs, node_id, output_path, start_frame=5, end_frame=824, 
                        dpi=300, title=None, xlabel='相对时间(s)', ylabel='实时速度'):
    """绘制多条曲线的对比图表
    
    参数:
        data_configs: 列表，每个元素是一个字典，包含以下键：
            - 'data': DataFrame，包含time和avg_speed列
            - 'label': str，图例标签
            - 'color': str，线条颜色
            - 'linestyle': str，线型 ('-', '--', '-.', ':')
            - 'linewidth': float，线宽（可选，默认1.5）
            - 'alpha': float，透明度（可选，默认1.0）
        node_id: int，节点ID
        output_path: str，输出文件路径
        start_frame: int，起始时间帧
        end_frame: int，结束时间帧
        dpi: int，图像分辨率
        title: str，图表标题（可选）
        xlabel: str，x轴标签
        ylabel: str，y轴标签
    """
    
    # 设置图表样式 - 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制每条曲线
    for config in data_configs:
        df = config['data']
        
        # 限制时间范围并按时间排序
        df = df[(df['time'] >= start_frame) & (df['time'] <= end_frame)].copy()
        df = df.sort_values('time')
        
        if len(df) == 0:
            print(f"警告：{config['label']} 数据为空，跳过绘制")
            continue
        
        # 获取绘图参数
        label = config['label']
        color = config['color']
        linestyle = config.get('linestyle', '-')
        linewidth = config.get('linewidth', 1.5)
        alpha = config.get('alpha', 1.0)
        
        # 绘制曲线
        ax.plot(df['time'].values, df['avg_speed'].values, 
                color=color, linestyle=linestyle, linewidth=linewidth, 
                alpha=alpha, label=label)
        
        print(f"绘制 {label}: {len(df)} 个点, 时间范围 {df['time'].min()}-{df['time'].max()}, "
              f"速度范围 {df['avg_speed'].min():.2f}-{df['avg_speed'].max():.2f}")
    
    # 设置图表属性
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title = f'速度特征变化曲线对比 (节点{node_id})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlim(start_frame, end_frame)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    plt.close()

def main():
    # ========== 配置参数 ==========
    # 节点ID（修改此处以分析不同节点）
    NODE_ID = 42
    
    # 时间范围
    start_frame = 250 #5
    end_frame = 600 #824
    
    # 输出路径
    output_path = f'../data/draw/d210191000/node{NODE_ID}_patrol_compare.png'
    
    # ========== 数据配置 ==========
    # 每条曲线的配置：文件路径、标签、颜色、线型等
    # 用户可以根据需要添加或修改曲线配置
    '''
        {
            'file_path': '../data/draw/d210191000/interpolated_data.csv',
            'label': '线性插值',
            'color': 'gray',
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 0.5,
            'value_column': 'avg_speed'  # 或 'interpolated_avg_speed'
        },
    '''
    curve_configs = [
        {
            'file_path': '../data/draw/d210191000/inference_results_15s.csv',
            'label': '15',
            'color': 'y',
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 0.5,
            'value_column': 'avg_speed'
        },
        {
            'file_path': '../data/draw/d210191000/inference_results_10s.csv',
            'label': '10',
            'color': 'blue',
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 0.5,
            'value_column': 'avg_speed'
        },
        {
            'file_path': '../data/draw/d210191000/inference_results_parabola.csv',
            'label': '抛物线',
            'color': 'gray',
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 0.5,
            'value_column': 'avg_speed'  # 或 'interpolated_avg_speed'
        },
        


        {
            'file_path': '../data/draw/d210191000/inference_results.csv',
            'label': '本研究模型',
            'color': 'black',
            'linestyle': '-',
            'linewidth': 1,
            'alpha': 1.0,
            'value_column': 'avg_speed'
        },
        {
            'file_path': '../data/draw/d210191000/d210291000_lane_node_stats.csv',
            'label': '真值',
            'color': 'red',
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 1.0,
            'value_column': 'avg_speed'  # 数据列名
        }
    ]
    
    # ========== 加载和处理数据 ==========
    print("加载数据...")
    data_configs = []
    
    for config in curve_configs:
        file_path = config['file_path']
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            print(f"加载 {file_path}: {len(df)} 条记录")
            
            # 筛选指定node_id的数据
            node_data = df[df['node_id'] == NODE_ID].copy()
            print(f"  筛选node_id={NODE_ID}: {len(node_data)} 条记录")
            
            if len(node_data) == 0:
                print(f"  警告：未找到node_id={NODE_ID}的数据，跳过此曲线")
                continue
            
            # 处理列名（支持不同的列名）
            value_column = config.get('value_column', 'avg_speed')
            
            # 如果指定的列不存在，尝试查找替代列
            if value_column not in node_data.columns:
                # 尝试常见的替代列名
                alternative_columns = ['interpolated_avg_speed', 'avg_speed', 'value']
                found = False
                for alt_col in alternative_columns:
                    if alt_col in node_data.columns:
                        print(f"  列 '{value_column}' 不存在，使用 '{alt_col}' 代替")
                        value_column = alt_col
                        found = True
                        break
                
                if not found:
                    print(f"  错误：未找到数据列，跳过此曲线")
                    print(f"  可用列: {node_data.columns.tolist()}")
                    continue
            
            # 重命名为统一的列名
            if value_column != 'avg_speed':
                node_data['avg_speed'] = node_data[value_column]
            
            # 添加到绘图配置
            data_configs.append({
                'data': node_data,
                'label': config['label'],
                'color': config['color'],
                'linestyle': config.get('linestyle', '-'),
                'linewidth': config.get('linewidth', 1.5),
                'alpha': config.get('alpha', 1.0)
            })
            
        except FileNotFoundError:
            print(f"警告：文件 {file_path} 不存在，跳过此曲线")
        except Exception as e:
            print(f"警告：处理文件 {file_path} 时出错: {e}")
    
    # ========== 绘制图表 ==========
    if len(data_configs) == 0:
        print("\n错误：没有可绘制的数据！")
        return
    
    print(f"\n绘制图表，共 {len(data_configs)} 条曲线...")
    plot_multiple_curves(
        data_configs=data_configs,
        node_id=NODE_ID,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        dpi=300,
        title=None,  # 使用默认标题
        xlabel='相对时间(s)',
        ylabel='实时速度'
    )
    
    print("\n完成!")

if __name__ == '__main__':
    main()
