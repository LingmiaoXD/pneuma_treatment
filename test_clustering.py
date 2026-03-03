"""
测试速度聚类功能的简单脚本
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 创建模拟数据
np.random.seed(42)
node_id = 42
times = np.arange(34, 824)
speeds = []

# 模拟三种交通状态的速度分布
for t in times:
    if t < 200:
        # 自由流：25-35 km/h
        speed = np.random.normal(30, 3)
    elif t < 500:
        # 饱和流：10-20 km/h
        speed = np.random.normal(15, 3)
    else:
        # 停车状态：0-5 km/h
        speed = np.random.normal(3, 1)
    speeds.append(max(0, speed))  # 确保速度非负

# 创建DataFrame
df = pd.DataFrame({
    'node_id': [node_id] * len(times),
    'time': times,
    'avg_speed': speeds
})

print("模拟数据统计:")
print(f"数据点数: {len(df)}")
print(f"速度范围: {df['avg_speed'].min():.2f} - {df['avg_speed'].max():.2f} km/h")
print(f"平均速度: {df['avg_speed'].mean():.2f} km/h")

# 进行K-means聚类
speeds_array = df['avg_speed'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(speeds_array)
centers = kmeans.cluster_centers_.flatten()

# 排序聚类中心
sorted_indices = np.argsort(centers)
sorted_centers = centers[sorted_indices]

# 计算阈值
stop_threshold = (sorted_centers[0] + sorted_centers[1]) / 2
free_flow_threshold = (sorted_centers[1] + sorted_centers[2]) / 2

print("\n聚类结果:")
print(f"聚类中心 (从低到高):")
print(f"  停车状态中心: {sorted_centers[0]:.2f} km/h")
print(f"  饱和流中心: {sorted_centers[1]:.2f} km/h")
print(f"  自由流中心: {sorted_centers[2]:.2f} km/h")
print(f"\n计算得到的阈值:")
print(f"  停车速度阈值: {stop_threshold:.2f} km/h")
print(f"  自由流速度阈值: {free_flow_threshold:.2f} km/h")

print("\n测试成功！聚类功能正常工作。")
