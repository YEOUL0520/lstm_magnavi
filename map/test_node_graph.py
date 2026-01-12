#!/usr/bin/env python3
"""
노드 그래프 기반 경로 계산 테스트
"""
import pandas as pd
import numpy as np
import networkx as nx

# 노드 정보 로드
nodes_df = pd.read_csv('nodes_final.csv')
print(f"총 노드: {len(nodes_df)}개\n")

# 노드 위치
node_positions = {row['id']: (row['x_m'], row['y_m']) for _, row in nodes_df.iterrows()}

# 그래프 생성 (가까운 노드들 연결)
G = nx.Graph()

# 모든 노드 추가
for node_id in node_positions:
    G.add_node(node_id)

# 가까운 노드들 연결 (거리 5m 이하)
CONNECTION_THRESHOLD = 5.0  # m

for node1, pos1 in node_positions.items():
    for node2, pos2 in node_positions.items():
        if node1 >= node2:
            continue

        # 거리 계산
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        if dist <= CONNECTION_THRESHOLD:
            G.add_edge(node1, node2, weight=dist)

print(f"연결된 엣지 수: {G.number_of_edges()}")

# 테스트: 1 → 23 경로
start = 1
end = 23

if nx.has_path(G, start, end):
    path = nx.shortest_path(G, start, end, weight='weight')
    print(f"\n경로 {start}→{end}:")
    print(f"  중간 노드: {path}")

    # 좌표 출력
    print(f"\n  좌표:")
    for node_id in path:
        x, y = node_positions[node_id]
        print(f"    노드 {node_id:2d}: ({x:6.1f}, {y:5.1f})")

    # X/Y 변화 확인
    print(f"\n  이동:")
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i+1]
        x1, y1 = node_positions[n1]
        x2, y2 = node_positions[n2]
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) > 0.01 and abs(dy) < 0.01:
            print(f"    {n1}→{n2}: X축 이동 ({dx:+.1f}m)")
        elif abs(dy) > 0.01 and abs(dx) < 0.01:
            print(f"    {n1}→{n2}: Y축 이동 ({dy:+.1f}m)")
        else:
            print(f"    {n1}→{n2}: 대각선? dx={dx:.1f}, dy={dy:.1f}")
else:
    print(f"\n경로 {start}→{end}: 연결 없음!")

# 다른 테스트 경로들
print("\n" + "="*70)
print("다른 경로 테스트:")

test_routes = [(1, 20), (8, 29), (10, 26), (15, 21)]

for start, end in test_routes:
    if nx.has_path(G, start, end):
        path = nx.shortest_path(G, start, end, weight='weight')
        print(f"\n{start}→{end}: {' → '.join(map(str, path))}")
    else:
        print(f"\n{start}→{end}: 연결 없음")
