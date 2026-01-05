#!/usr/bin/env python3
"""노드 연결 관계 자동 생성"""
import csv
from pathlib import Path

def generate_connections(nodes_path: Path, output_path: Path, max_distance: float = 6.0):
    """노드 위치 기반으로 연결 관계 생성"""

    # 노드 읽기
    positions = {}
    with nodes_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["id"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            positions[node_id] = (x, y)

    # 연결 관계 생성
    connections = set()
    nodes = sorted(positions.keys())

    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            xa, ya = positions[a]
            xb, yb = positions[b]

            # Manhattan distance
            dist = abs(xb - xa) + abs(yb - ya)

            # max_distance 이하이고 같은 행/열인 경우만 연결
            if dist <= max_distance:
                same_row = abs(ya - yb) < 0.5  # y 차이 < 0.5m
                same_col = abs(xa - xb) < 0.5  # x 차이 < 0.5m

                if same_row or same_col:
                    # 작은 노드 번호가 먼저 오도록
                    if a < b:
                        connections.add((a, b))
                    else:
                        connections.add((b, a))

    # CSV로 저장
    with output_path.open('w') as f:
        f.write("node1,node2\n")
        for a, b in sorted(connections):
            f.write(f"{a},{b}\n")

    print(f"✅ {len(connections)}개 연결 생성: {output_path}")

    # 연결 통계
    node_degree = {}
    for a, b in connections:
        node_degree[a] = node_degree.get(a, 0) + 1
        node_degree[b] = node_degree.get(b, 0) + 1

    print("\n노드별 연결 수:")
    for node in sorted(node_degree.keys()):
        print(f"  노드 {node}: {node_degree[node]}개 연결")

    # 연결 안 된 노드 확인
    unconnected = set(nodes) - set(node_degree.keys())
    if unconnected:
        print(f"\n⚠️  연결 안 된 노드: {sorted(unconnected)}")

if __name__ == "__main__":
    generate_connections(
        Path("data/nodes_final.csv"),
        Path("data/node_connections.csv"),
        max_distance=6.0
    )
