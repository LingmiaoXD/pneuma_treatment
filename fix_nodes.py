import json

# 读取原始文件
with open('data/road_graph/minhang_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 从 lanes 中提取每个 lane 的 nodes 列表
lane_nodes_map = {}
for lane in data['lanes']:
    lane_id = lane['lane_id']
    nodes = lane['nodes']
    lane_nodes_map[lane_id] = nodes

# 生成新的 nodes 数组
new_nodes = []
for lane_id, node_ids in sorted(lane_nodes_map.items()):
    for node_id in node_ids:
        new_node = {
            "node_id": node_id,
            "lane_id": lane_id,
            "position_in_lane": None,
            "segment_length": 10.0,
            "node_connections": {
                "direct": [],
                "near": []
            }
        }
        new_nodes.append(new_node)

# 替换原有的 nodes
data['nodes'] = new_nodes

# 保存到新文件
with open('data/road_graph/minhang_graph_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已生成 {len(new_nodes)} 个节点")
print("新文件已保存到: data/road_graph/minhang_graph_fixed.json")
