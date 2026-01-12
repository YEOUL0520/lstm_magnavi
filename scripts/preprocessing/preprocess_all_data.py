#!/usr/bin/env python3
"""모든 raw 데이터 전처리"""
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 격자 간격
GRID_DISTANCE = 0.45

def read_nodes(path: Path) -> Dict[int, Tuple[float, float]]:
    """노드 위치 읽기"""
    positions = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["id"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            positions[node_id] = (x, y)
    return positions

def build_graph(positions: Dict[int, Tuple[float, float]], connections_path: Path) -> Dict[int, List[Tuple[int, float]]]:
    """그래프 구축"""
    graph = {node: [] for node in positions}
    with connections_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["node1"])
            b = int(row["node2"])
            if a not in positions or b not in positions:
                continue
            xa, ya = positions[a]
            xb, yb = positions[b]
            dist = abs(xb - xa) + abs(yb - ya)
            graph[a].append((b, dist))
            graph[b].append((a, dist))
    return graph

def find_shortest_path(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> List[int]:
    """BFS로 최단 경로 찾기"""
    if start == end:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for neighbor, _ in graph[node]:
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == end:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    return None

def interpolate_grid_on_path(
    grid_idx: int,
    path: List[int],
    positions: Dict[int, Tuple[float, float]],
    grid_distance: float = GRID_DISTANCE
) -> Tuple[float, float]:
    """격자 인덱스를 BFS 경로 상의 좌표로 변환"""
    segments = []
    for i in range(len(path) - 1):
        p1 = positions[path[i]]
        p2 = positions[path[i+1]]
        dist = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
        segments.append({'start_pos': p1, 'end_pos': p2, 'distance': dist})

    total_length = sum(seg['distance'] for seg in segments)
    cumulative_dist = grid_idx * grid_distance

    if cumulative_dist >= total_length:
        return positions[path[-1]]

    cumulative = 0
    for seg in segments:
        if cumulative + seg['distance'] >= cumulative_dist:
            seg_progress = (cumulative_dist - cumulative) / seg['distance'] if seg['distance'] > 0 else 0.0
            p1 = seg['start_pos']
            p2 = seg['end_pos']
            x = p1[0] + seg_progress * (p2[0] - p1[0])
            y = p1[1] + seg_progress * (p2[1] - p1[1])
            return (x, y)
        cumulative += seg['distance']

    return positions[path[-1]]

def preprocess_file(file_path: Path, positions: Dict, graph: Dict, output_dir: Path) -> Dict:
    """파일 하나 전처리"""
    parts = file_path.stem.split("_")
    if len(parts) < 3:
        return {'status': 'error', 'message': '파일명 형식 오류'}

    start_node = int(parts[0])
    end_node = int(parts[1])
    session_id = parts[2]

    # 캐싱: 이미 처리된 파일이 있고 더 최신이면 스킵
    output_file = output_dir / f"{start_node}_{end_node}_{session_id}.csv"
    if output_file.exists():
        if output_file.stat().st_mtime >= file_path.stat().st_mtime:
            return {'status': 'cached', 'message': 'Already processed'}

    path = find_shortest_path(graph, start_node, end_node)
    if path is None:
        return {'status': 'error', 'message': f'경로 없음: {start_node} → {end_node}'}

    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return {'status': 'error', 'message': '빈 파일'}

    # 1단계: 격자 위치 찾기
    grid_rows = []
    grid_index = 0

    for i, row in enumerate(rows):
        highlighted = row.get("Highlighted", "false").lower() == "true"
        rightangle = row.get("RightAngle", "false").lower() == "true"

        if highlighted or rightangle:
            x, y = interpolate_grid_on_path(grid_index, path, positions, GRID_DISTANCE)
            grid_rows.append((i, grid_index, x, y))
            grid_index += 1

    if len(grid_rows) == 0:
        return {'status': 'error', 'message': '격자 없음'}

    # 2단계: 첫 격자부터 마지막 격자까지만 선형 보간
    first_grid_idx = grid_rows[0][0]
    last_grid_idx = grid_rows[-1][0]

    output_rows = []

    for i in range(first_grid_idx, last_grid_idx + 1):
        row = rows[i]

        # 현재 행 이전/이후 격자 찾기
        prev_grid = None
        next_grid = None

        for g in grid_rows:
            if g[0] <= i:
                prev_grid = g
            if g[0] >= i and next_grid is None:
                next_grid = g

        # 선형 보간
        if prev_grid and next_grid and prev_grid[0] != next_grid[0]:
            progress = (i - prev_grid[0]) / (next_grid[0] - prev_grid[0])
            x = prev_grid[2] + progress * (next_grid[2] - prev_grid[2])
            y = prev_grid[3] + progress * (next_grid[3] - prev_grid[3])
        elif prev_grid:
            x, y = prev_grid[2], prev_grid[3]
        elif next_grid:
            x, y = next_grid[2], next_grid[3]
        else:
            x, y = positions[path[0]]

        output_row = {
            'x': x,
            'y': y,
            'magx': row['MagX'],
            'magy': row['MagY'],
            'magz': row['MagZ'],
            'yaw': row['Yaw'],
            'roll': row['Roll'],
            'pitch': row['Pitch'],
            'timestamp': row['Timestamp'],
            'accx': row['AccX'],
            'accy': row['AccY'],
            'accz': row['AccZ'],
            'gyrox': row['GyroX'],
            'gyroy': row['GyroY'],
            'gyroz': row['GyroZ'],
            'highlighted': row['Highlighted'],
            'rightangle': row['RightAngle'],
        }
        output_rows.append(output_row)

    with output_file.open('w', newline='') as f:
        fieldnames = ['x', 'y', 'magx', 'magy', 'magz', 'yaw', 'roll', 'pitch',
                      'timestamp', 'accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz',
                      'highlighted', 'rightangle']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return {'status': 'success', 'total_rows': len(output_rows)}

def preprocess_file_wrapper(args):
    """멀티프로세싱용 래퍼"""
    file_path, positions, graph, output_dir = args
    return preprocess_file(file_path, positions, graph, output_dir)

def main():
    nodes_path = Path("data/nodes_final.csv")
    connections_path = Path("data/node_connections.csv")
    raw_dir = Path("data/raw")
    output_dir = Path("data/preprocessed")

    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("Raw → Preprocessed 전처리 시작")
    print("=" * 80)

    print("노드 및 그래프 로딩...")
    positions = read_nodes(nodes_path)
    graph = build_graph(positions, connections_path)

    raw_files = sorted(raw_dir.glob("*.csv"))
    print(f"총 {len(raw_files)}개 파일 발견\n")

    # 멀티프로세싱 설정
    n_workers = min(cpu_count(), 8)
    print(f"병렬 처리: {n_workers} workers\n")

    # 인자 준비
    args_list = [(f, positions, graph, output_dir) for f in raw_files]

    # 병렬 처리
    success_count = 0
    error_count = 0
    cached_count = 0
    total_rows = 0

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(preprocess_file_wrapper, args_list),
            total=len(raw_files),
            desc="전처리 중",
            ncols=80,
            unit="file"
        ))

    # 결과 집계
    for result in results:
        if result['status'] == 'success':
            success_count += 1
            total_rows += result['total_rows']
        elif result['status'] == 'cached':
            cached_count += 1
        else:
            error_count += 1

    print()
    print("=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)
    print(f"  성공:   {success_count}개 (새로 처리)")
    print(f"  캐싱:   {cached_count}개 (이미 처리됨)")
    print(f"  실패:   {error_count}개")
    print(f"  총 행:  {total_rows:,}행")
    print(f"  출력:   {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
