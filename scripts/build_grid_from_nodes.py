#!/usr/bin/env python3
"""Generate grid-level nodes/connections at 0.45m spacing from aligned segments."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

GRID_DISTANCE = 0.45
TOL = 1e-6


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def rounded(self, ndigits: int = 6) -> "Point":
        return Point(round(self.x, ndigits), round(self.y, ndigits))


@dataclass
class NodeInfo:
    id: int
    name: str
    point: Point
    type: str


def read_nodes(path: Path) -> Dict[int, NodeInfo]:
    nodes: Dict[int, NodeInfo] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["id"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            name = row.get("name", str(node_id))
            ntype = row.get("type", "marker")
            nodes[node_id] = NodeInfo(node_id, name, Point(x, y), ntype)
    return nodes


def read_connections(path: Path) -> List[Tuple[int, int]]:
    connections: List[Tuple[int, int]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["node1"])
            b = int(row["node2"])
            connections.append((a, b))
    return connections


def is_axis_aligned(pa: Point, pb: Point) -> bool:
    return abs(pa.x - pb.x) < TOL or abs(pa.y - pb.y) < TOL


def is_multiple_of_grid(dist: float) -> bool:
    ratio = dist / GRID_DISTANCE
    return abs(ratio - round(ratio)) < 1e-6


def iter_segment_points(pa: Point, pb: Point) -> Iterable[Point]:
    """Yield points (inclusive of endpoints) every GRID_DISTANCE along an axis-aligned segment."""
    if not is_axis_aligned(pa, pb):
        return []

    dx = pb.x - pa.x
    dy = pb.y - pa.y
    dist = abs(dx) + abs(dy)
    steps = int(round(dist / GRID_DISTANCE))

    step_x = 0.0 if abs(dx) < TOL else (GRID_DISTANCE if dx > 0 else -GRID_DISTANCE)
    step_y = 0.0 if abs(dy) < TOL else (GRID_DISTANCE if dy > 0 else -GRID_DISTANCE)

    return (
        Point(pa.x + step_x * i, pa.y + step_y * i).rounded()
        for i in range(steps + 1)
    )


def build_grid(nodes: Dict[int, NodeInfo], connections: List[Tuple[int, int]]):
    # Seed with all existing nodes (turn/marker) so turn은 유지 and ids remain.
    grid_points: Dict[Point, int] = {}
    node_info: Dict[int, NodeInfo] = {}
    for nid, info in nodes.items():
        p = info.point.rounded()
        grid_points[p] = nid
        seeded_type = "turn" if info.type == "turn" else "grid"
        node_info[nid] = NodeInfo(nid, info.name, p, seeded_type)

    grid_edges: set[Tuple[int, int]] = set()
    next_id = max(nodes.keys()) + 1 if nodes else 1

    for a, b in connections:
        if a not in nodes or b not in nodes:
            continue

        pa, pb = nodes[a].point, nodes[b].point
        dist = abs(pa.x - pb.x) + abs(pa.y - pb.y)

        if not is_axis_aligned(pa, pb):
            continue
        if not is_multiple_of_grid(dist):
            continue

        segment_point_ids: List[int] = []
        for p in iter_segment_points(pa, pb):
            p = p.rounded()
            if p not in grid_points:
                grid_points[p] = next_id
                node_info[next_id] = NodeInfo(next_id, str(next_id), p, "grid")
                next_id += 1
            segment_point_ids.append(grid_points[p])

        # connect consecutive points along this segment
        for i in range(len(segment_point_ids) - 1):
            n1, n2 = segment_point_ids[i], segment_point_ids[i + 1]
            edge = (min(n1, n2), max(n1, n2))
            grid_edges.add(edge)

    return node_info, grid_edges


def apply_existing_types(node_info: Dict[int, NodeInfo], original: Dict[int, NodeInfo]) -> None:
    """If a grid node position matches an original turn, inherit its type."""
    turn_by_point: Dict[Point, str] = {}
    for info in original.values():
        if info.type == "turn":
            turn_by_point[info.point.rounded()] = info.type

    for info in node_info.values():
        if info.point in turn_by_point:
            info.type = turn_by_point[info.point]


def write_grid(nodes: Dict[int, NodeInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "x_m", "y_m", "type"])
        for pid in sorted(nodes.keys()):
            node = nodes[pid]
            writer.writerow([node.id, node.name, node.point.x, node.point.y, node.type])


def write_edges(edges: Iterable[Tuple[int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node1", "node2"])
        for a, b in sorted(edges):
            writer.writerow([a, b])


def main():
    nodes = read_nodes(Path("data/nodes_final.csv"))
    connections = read_connections(Path("data/node_connections.csv"))

    node_info, edges = build_grid(nodes, connections)
    apply_existing_types(node_info, nodes)

    write_grid(node_info, Path("data/grid_final.csv"))
    write_edges(edges, Path("data/grid_connections.csv"))

    print(f"✅ grid nodes (including original turns/markers): {len(node_info)}, grid connections: {len(edges)}")


if __name__ == "__main__":
    main()
