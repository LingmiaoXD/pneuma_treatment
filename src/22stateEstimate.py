'''
    输入：
        多个CSV文件，每个文件包含节点的时间序列数据，代表：真值、本模型的预测值、其他基准模型的预测值

    思路：
        从多个CSV文件中读取指定node_id的数据
        按照时间顺序排列，对每个时间节点所处的交通状态进行判断：
        如果avg_speed>25或者为空，则为自由流，绿色
        如果avg_speed处于5到25之间，且后10s的趋势存在明显下降，则为排队形成，橙色
        如果avg_speed处于5到25之间，且后10s的趋势存在明显上升，则为排队消散，淡蓝色
        如果avg_speed处于5到25之间，且后10s不稳定，则为饱和流，黄色
        如果avg_speed处于5及以下，则为排队状态，红色
        每个文件对应node_id，对应一条横着的条状图，表示随着时间状态的变化

    输出：
        一张高分辨率的对比图表
        从上到下为多个文件对应的同一node_id表示状态变化时间线的条状图
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 定义交通状态
class TrafficState:
    FREE_FLOW = 0          # 自由流
    QUEUE_FORMING = 1      # 排队形成
    QUEUE_DISSIPATING = 2  # 排队消散
    SATURATED = 3          # 饱和流
    QUEUED = 4             # 排队状态
    UNKNOWN = 5            # 未知（无数据）

# 状态颜色映射
STATE_COLORS = {
    TrafficState.FREE_FLOW: '#2ecc71',        # 绿色
    TrafficState.QUEUE_FORMING: '#e67e22',    # 橙色
    TrafficState.QUEUE_DISSIPATING: '#3498db', # 淡蓝色
    TrafficState.SATURATED: '#f1c40f',        # 黄色
    TrafficState.QUEUED: '#e74c3c',           # 红色
    TrafficState.UNKNOWN: '#95a5a6'           # 灰色
}

# 状态名称映射
STATE_NAMES = {
    TrafficState.FREE_FLOW: '自由流',
    TrafficState.QUEUE_FORMING: '排队形成',
    TrafficState.QUEUE_DISSIPATING: '排队消散',
    TrafficState.SATURATED: '饱和流',
    TrafficState.QUEUED: '排队状态',
    TrafficState.UNKNOWN: '无数据'
}

def calculate_trend(speeds, window=10):
    """计算速度趋势
    
    参数:
        speeds: 速度序列（未来window个时间点）
        window: 趋势计算窗口大小
    
    返回:
        'increasing': 明显上升
        'decreasing': 明显下降
        'unstable': 不稳定
        'stable': 稳定
    """
    if len(speeds) < 3:
        return 'stable'
    
    # 计算线性回归斜率
    x = np.arange(len(speeds))
    y = np.array(speeds)
    
    # 去除NaN值
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 3:
        return 'stable'
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # 线性拟合
    slope = np.polyfit(x_valid, y_valid, 1)[0]
    
    # 计算标准差（波动性）
    std = np.std(y_valid)
    
    # 判断趋势
    if abs(slope) < 0.5:  # 斜率很小，认为稳定
        if std > 3:  # 但波动大
            return 'unstable'
        return 'stable'
    elif slope > 1.5:  # 明显上升
        return 'increasing'
    elif slope < -1.5:  # 明显下降
        return 'decreasing'
    else:
        if std > 3:
            return 'unstable'
        return 'stable'

def classify_traffic_state(speed, future_speeds, lower_threshold=5, upper_threshold=25):
    """分类交通状态
    
    参数:
        speed: 当前速度
        future_speeds: 未来10s的速度序列
        lower_threshold: 下限阈值，默认为5
        upper_threshold: 上限阈值，默认为25
    
    返回:
        TrafficState枚举值
    """
    # 无数据或速度为NaN，默认为自由流
    if pd.isna(speed):
        return TrafficState.FREE_FLOW
    
    # 自由流：速度>upper_threshold
    if speed > upper_threshold:
        return TrafficState.FREE_FLOW
    
    # 排队状态：速度<=lower_threshold
    if speed <= lower_threshold:
        return TrafficState.QUEUED
    
    # 速度在lower_threshold-upper_threshold之间，需要判断趋势
    trend = calculate_trend(future_speeds)
    
    if trend == 'decreasing':
        return TrafficState.QUEUE_FORMING
    elif trend == 'increasing':
        return TrafficState.QUEUE_DISSIPATING
    else:  # unstable or stable
        return TrafficState.SATURATED

def estimate_states(df, node_id, value_column='avg_speed', window=10, 
                   lower_threshold=5, upper_threshold=25):
    """估计时间序列的交通状态
    
    参数:
        df: DataFrame，包含time和速度列
        node_id: 节点ID
        value_column: 速度列名
        window: 趋势计算窗口大小
        lower_threshold: 下限阈值，默认为5
        upper_threshold: 上限阈值，默认为25
    
    返回:
        DataFrame，包含time和state列
    """
    # 筛选节点数据并排序
    node_data = df[df['node_id'] == node_id].copy()
    node_data = node_data.sort_values('time')
    
    if len(node_data) == 0:
        return pd.DataFrame(columns=['time', 'state'])
    
    # 确保速度列存在
    if value_column not in node_data.columns:
        print(f"警告：列 '{value_column}' 不存在")
        return pd.DataFrame(columns=['time', 'state'])
    
    # 重命名为统一列名
    node_data['speed'] = node_data[value_column]
    
    # 计算每个时间点的状态
    states = []
    times = node_data['time'].values
    speeds = node_data['speed'].values
    
    for i in range(len(times)):
        current_speed = speeds[i]
        
        # 获取未来window个时间点的速度
        future_speeds = speeds[i:min(i+window+1, len(speeds))]
        
        # 分类状态
        state = classify_traffic_state(current_speed, future_speeds, 
                                      lower_threshold, upper_threshold)
        states.append(state)
    
    result = pd.DataFrame({
        'time': times,
        'state': states
    })
    
    return result

def plot_state_timeline(state_configs, node_id, output_path, start_frame=5, end_frame=824,
                       dpi=300, figsize=(14, 8), title=None):
    """绘制交通状态时间线对比图
    
    参数:
        state_configs: 列表，每个元素是一个字典，包含：
            - 'states': DataFrame，包含time和state列
            - 'label': str，标签
        node_id: 节点ID
        output_path: 输出路径
        start_frame: 起始时间
        end_frame: 结束时间
        dpi: 图像分辨率
        figsize: 图像大小
        title: 图表标题
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_configs = len(state_configs)
    fig, ax = plt.subplots(figsize=figsize)
    
    # 每个配置占据的高度
    bar_height = 0.4  # 从0.8调整为0.4，缩小到一半
    bar_spacing = 0.2
    
    # 绘制每个配置的状态条
    for idx, config in enumerate(state_configs):
        states_df = config['states']
        label = config['label']
        
        # 限制时间范围
        states_df = states_df[(states_df['time'] >= start_frame) & 
                             (states_df['time'] <= end_frame)].copy()
        
        if len(states_df) == 0:
            print(f"警告：{label} 无数据，跳过")
            continue
        
        # y位置（从上到下）
        y_pos = n_configs - idx - 1
        
        # 绘制状态段
        for i in range(len(states_df)):
            time = states_df.iloc[i]['time']
            state = states_df.iloc[i]['state']
            
            # 确定时间段宽度（到下一个时间点）
            if i < len(states_df) - 1:
                width = states_df.iloc[i+1]['time'] - time
            else:
                width = 1  # 最后一个点默认宽度为1
            
            # 绘制矩形
            color = STATE_COLORS[state]
            rect = mpatches.Rectangle((time, y_pos - bar_height/2), width, bar_height,
                                     facecolor=color, edgecolor='none')
            ax.add_patch(rect)
        
        # 添加标签
        ax.text(start_frame - (end_frame - start_frame) * 0.02, y_pos, label,
               ha='right', va='center', fontsize=11, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xlim(start_frame, end_frame)
    ax.set_ylim(-0.5, n_configs - 0.5)
    ax.set_xlabel('相对时间(s)', fontsize=12)
    ax.set_yticks([])
    
    if title is None:
        title = f'交通状态时间线对比 (节点{node_id})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=STATE_COLORS[state], 
                                     label=STATE_NAMES[state])
                      for state in [TrafficState.FREE_FLOW, 
                                   TrafficState.QUEUE_FORMING,
                                   TrafficState.QUEUE_DISSIPATING,
                                   TrafficState.SATURATED,
                                   TrafficState.QUEUED,
                                   TrafficState.UNKNOWN]]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.15, 1), fontsize=10)
    
    # 网格
    ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    plt.close()

def main():
    # ========== 配置参数 ==========
    NODE_ID = 42
    start_frame = 34   #34
    end_frame = 824     #824
    output_path = f'../data/draw/d210191000/node{NODE_ID}_state_timeline.png'
    
    # ========== 数据配置 ==========
    data_configs = [
        {
            'file_path': '../data/draw/d210191000/d210291000_lane_node_stats.csv',
            'label': '真值',
            'value_column': 'avg_speed',
            'lower_threshold': 5,   # 真值使用5-25
            'upper_threshold': 25
        },
        {
            'file_path': '../data/draw/d210191000/inference_results.csv',
            'label': '本研究模型',
            'value_column': 'avg_speed',
            'lower_threshold': 5,   # 本研究模型使用5-25
            'upper_threshold': 25
        },
        {
            'file_path': '../data/draw/d210191000/simple_stgnn_predictions.csv',
            'label': 'STGNN',
            'value_column': 'avg_speed',
            'lower_threshold': 5,   # 其他模型使用5-20
            'upper_threshold': 25  # 其他模型使用20
        },
        {
            'file_path': '../data/draw/d210191000/physical_prior_predictions.csv',
            'label': '物理模型法',
            'value_column': 'avg_speed',
            'lower_threshold': 5,   # 其他模型使用5-20
            'upper_threshold': 25  # 其他模型使用20
        },
        {
            'file_path': '../data/draw/d210191000/phase_template_results.csv',
            'label': '模板法',
            'value_column': 'avg_speed',
            'lower_threshold': 5,   # 其他模型使用5-20
            'upper_threshold': 25  # 其他模型使用20
        },
        {
            'file_path': '../data/draw/d210191000/st_idw_results.csv',
            'label': 'ST-IDW',
            'value_column': 'avg_speed',
            'lower_threshold': 5,   # 其他模型使用5-20
            'upper_threshold': 23  # 其他模型使用20
        }
    ]
    
    # ========== 加载数据并估计状态 ==========
    print("加载数据并估计交通状态...")
    state_configs = []
    
    for config in data_configs:
        file_path = config['file_path']
        label = config['label']
        value_column = config.get('value_column', 'avg_speed')
        lower_threshold = config.get('lower_threshold', 5)   # 默认使用5
        upper_threshold = config.get('upper_threshold', 25)  # 默认使用25
        
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            print(f"\n加载 {label}: {len(df)} 条记录 (阈值: {lower_threshold}-{upper_threshold})")
            
            # 估计状态
            states_df = estimate_states(df, NODE_ID, value_column=value_column, 
                                       lower_threshold=lower_threshold,
                                       upper_threshold=upper_threshold)
            
            if len(states_df) == 0:
                print(f"  警告：未找到node_id={NODE_ID}的数据，跳过")
                continue
            
            print(f"  估计状态: {len(states_df)} 个时间点")
            
            # 统计各状态数量
            state_counts = states_df['state'].value_counts()
            for state, count in state_counts.items():
                print(f"    {STATE_NAMES[state]}: {count} ({count/len(states_df)*100:.1f}%)")
            
            state_configs.append({
                'states': states_df,
                'label': label
            })
            
        except FileNotFoundError:
            print(f"警告：文件 {file_path} 不存在，跳过")
        except Exception as e:
            print(f"警告：处理文件 {file_path} 时出错: {e}")
    
    # ========== 绘制图表 ==========
    if len(state_configs) == 0:
        print("\n错误：没有可绘制的数据！")
        return
    
    print(f"\n绘制状态时间线图表，共 {len(state_configs)} 条时间线...")
    plot_state_timeline(
        state_configs=state_configs,
        node_id=NODE_ID,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        dpi=300,
        figsize=(14, len(state_configs) * 1.5 + 2)
    )
    
    print("\n完成!")

if __name__ == '__main__':
    main()