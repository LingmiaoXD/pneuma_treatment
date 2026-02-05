'''
输入：
    方向级patrol_mask文件
    图结构文件，参考minhang_graph.json
    dynamic文件，参考d210240930_lane_node_stats.csv

处理：
    根据方向级patrol_mask文件，查找哪些时段、哪些方向是可见的
    依据图结构文件，查找每个时段可见方向对应的道路，再由道路查找节点
    如果一个节点

输出：
'''