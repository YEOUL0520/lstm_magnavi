#!/usr/bin/env python3
"""
노드 그래프 맵 그리기
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 노드 정보 로드
nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m']) for _, row in nodes_df.iterrows()}

# 그래프 생성
G = nx.Graph()
for node_id in node_positions:
    G.add_node(node_id)

# 가까운 노드들 연결 (5m 이하)
CONNECTION_THRESHOLD = 5.0
for node1, pos1 in node_positions.items():
    for node2, pos2 in node_positions.items():
        if node1 >= node2:
            continue
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        if dist <= CONNECTION_THRESHOLD:
            G.add_edge(node1, node2, weight=dist)

# 잘못된 연결 제거 (실제로는 연결되지 않음)
wrong_connections = [(10, 28), (24, 25)]
for n1, n2 in wrong_connections:
    if G.has_edge(n1, n2):
        G.remove_edge(n1, n2)
        print(f"  연결 제거: {n1}-{n2}")

# 시각화
fig, ax = plt.subplots(figsize=(16, 8))

# 노드 그리기
for node_id, (x, y) in node_positions.items():
    ax.plot(x, y, 'o', markersize=10, color='blue')
    ax.text(x, y + 0.5, str(node_id), fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 엣지 그리기
for node1, node2 in G.edges():
    x1, y1 = node_positions[node1]
    x2, y2 = node_positions[node2]
    ax.plot([x1, x2], [y1, y2], 'g-', linewidth=1, alpha=0.5)

# 예시 경로 강조 (1→23)
if nx.has_path(G, 1, 23):
    path = nx.shortest_path(G, 1, 23, weight='weight')
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i+1]
        x1, y1 = node_positions[n1]
        x2, y2 = node_positions[n2]
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=0.8, label='Path 1→23' if i == 0 else '')

ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title('Node Graph Map (Connection Threshold: 5.0m)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('node_graph_map.png', dpi=150, bbox_inches='tight')
print("✅ 지도 저장: node_graph_map.png")

# 노드 정보 출력
print("\n노드 정보:")
for node_id in sorted(node_positions.keys()):
    x, y = node_positions[node_id]
    degree = G.degree(node_id)
    print(f"  노드 {node_id:2d}: ({x:6.1f}, {y:5.1f}) - 연결: {degree}개")
