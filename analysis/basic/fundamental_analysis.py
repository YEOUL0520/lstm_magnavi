#!/usr/bin/env python3
"""근본적 분석: Bad vs Raw 차이의 진짜 원인"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

bad_dir = Path("data/bad")
raw_dir = Path("data/raw")
nodes_path = Path("data/nodes_final.csv")

# 노드 위치 읽기
node_positions = {}
with nodes_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_positions[int(row["id"])] = (float(row["x_m"]), float(row["y_m"]))

def analyze_file_full(file_path):
    """파일의 모든 센서 값 분석"""
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    try:
        result = {
            "filename": file_path.name,
            "length": len(rows),
            "timestamp_first": rows[0].get("Timestamp", ""),
            "timestamp_last": rows[-1].get("Timestamp", ""),
            "magx": np.array([float(row["MagX"]) for row in rows]),
            "magy": np.array([float(row["MagY"]) for row in rows]),
            "magz": np.array([float(row["MagZ"]) for row in rows]),
            "pitch": np.array([float(row["Pitch"]) for row in rows]),
            "roll": np.array([float(row["Roll"]) for row in rows]),
            "yaw": np.array([float(row["Yaw"]) for row in rows]),
        }

        # 통계
        result["magx_mean"] = np.mean(result["magx"])
        result["magy_mean"] = np.mean(result["magy"])
        result["magz_mean"] = np.mean(result["magz"])
        result["magx_std"] = np.std(result["magx"])
        result["magy_std"] = np.std(result["magy"])
        result["magz_std"] = np.std(result["magz"])

        # 경로 정보
        parts = file_path.stem.split("_")
        if len(parts) >= 2:
            result["start_node"] = int(parts[0])
            result["end_node"] = int(parts[1])
            result["path"] = f"{parts[0]}->{parts[1]}"

            # 공간 정보
            if result["start_node"] in node_positions and result["end_node"] in node_positions:
                start_pos = node_positions[result["start_node"]]
                end_pos = node_positions[result["end_node"]]
                result["start_x"] = start_pos[0]
                result["start_y"] = start_pos[1]
                result["end_x"] = end_pos[0]
                result["end_y"] = end_pos[1]
                result["center_x"] = (start_pos[0] + end_pos[0]) / 2
                result["center_y"] = (start_pos[1] + end_pos[1]) / 2

        return result
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

print("=" * 100)
print("🔬 근본적 분석: Bad vs Raw 차이의 진짜 원인")
print("=" * 100)
print()

# 데이터 수집
print("📊 전체 데이터 분석 중...")
bad_data = []
raw_data = []

for f in bad_dir.glob("*.csv"):
    result = analyze_file_full(f)
    if result:
        bad_data.append(result)

for f in raw_dir.glob("*.csv"):
    result = analyze_file_full(f)
    if result:
        raw_data.append(result)

print(f"Bad: {len(bad_data)}개 파일 분석 완료")
print(f"Raw: {len(raw_data)}개 파일 분석 완료")
print()

# ============================================================================
# 1. 시간 분석
# ============================================================================
print("=" * 100)
print("1. 시간 분석 (언제 측정했는가?)")
print("=" * 100)

def parse_timestamp(ts):
    """타임스탬프 파싱"""
    try:
        # 예: "2025-09-19T19:31:38.830694"
        return datetime.fromisoformat(ts.replace('Z', '+00:00').split('.')[0])
    except:
        return None

bad_dates = []
raw_dates = []

for d in bad_data:
    ts = parse_timestamp(d["timestamp_first"])
    if ts:
        bad_dates.append(ts)

for d in raw_data:
    ts = parse_timestamp(d["timestamp_first"])
    if ts:
        raw_dates.append(ts)

if bad_dates and raw_dates:
    print(f"\nBad 측정 기간:")
    print(f"  최초: {min(bad_dates)}")
    print(f"  최종: {max(bad_dates)}")
    print(f"  기간: {(max(bad_dates) - min(bad_dates)).days}일")

    print(f"\nRaw 측정 기간:")
    print(f"  최초: {min(raw_dates)}")
    print(f"  최종: {max(raw_dates)}")
    print(f"  기간: {(max(raw_dates) - min(raw_dates)).days}일")

    # 겹치는 날짜
    bad_days = set(d.date() for d in bad_dates)
    raw_days = set(d.date() for d in raw_dates)
    overlap = bad_days & raw_days

    print(f"\n겹치는 날짜: {len(overlap)}일")

    if len(overlap) == 0:
        print("\n🎯 결론: **완전히 다른 시간에 측정**")
        print("  → Bad와 Raw는 서로 다른 측정 세션")
    elif len(overlap) > 10:
        print("\n🎯 결론: **같은 기간에 측정**")
        print("  → 센서 차이일 가능성")
    else:
        print("\n🎯 결론: **일부 겹침**")
else:
    print("\n⚠️  타임스탬프 파싱 실패")

# ============================================================================
# 2. 공간 분석
# ============================================================================
print("\n" + "=" * 100)
print("2. 공간 분석 (어느 지역을 측정했는가?)")
print("=" * 100)

bad_with_pos = [d for d in bad_data if "center_x" in d]
raw_with_pos = [d for d in raw_data if "center_x" in d]

if bad_with_pos and raw_with_pos:
    bad_x = [d["center_x"] for d in bad_with_pos]
    bad_y = [d["center_y"] for d in bad_with_pos]
    raw_x = [d["center_x"] for d in raw_with_pos]
    raw_y = [d["center_y"] for d in raw_with_pos]

    print(f"\nBad 경로 중심 위치:")
    print(f"  X 범위: {min(bad_x):.1f} ~ {max(bad_x):.1f} (평균 {np.mean(bad_x):.1f})")
    print(f"  Y 범위: {min(bad_y):.1f} ~ {max(bad_y):.1f} (평균 {np.mean(bad_y):.1f})")

    print(f"\nRaw 경로 중심 위치:")
    print(f"  X 범위: {min(raw_x):.1f} ~ {max(raw_x):.1f} (평균 {np.mean(raw_x):.1f})")
    print(f"  Y 범위: {min(raw_y):.1f} ~ {max(raw_y):.1f} (평균 {np.mean(raw_y):.1f})")

    # 공간적 분리도
    bad_center = (np.mean(bad_x), np.mean(bad_y))
    raw_center = (np.mean(raw_x), np.mean(raw_y))
    distance = np.sqrt((bad_center[0] - raw_center[0])**2 + (bad_center[1] - raw_center[1])**2)

    print(f"\n중심간 거리: {distance:.1f}m")

    if distance > 20:
        print("\n🎯 결론: **공간적으로 분리됨**")
        print("  → Bad와 Raw는 다른 지역 측정")
    else:
        print("\n🎯 결론: **같은 지역**")

# ============================================================================
# 3. 모든 센서 비교
# ============================================================================
print("\n" + "=" * 100)
print("3. 모든 센서 비교 (MagX만 다른가?)")
print("=" * 100)

sensors = ["magx", "magy", "magz"]

for sensor in sensors:
    bad_means = [d[f"{sensor}_mean"] for d in bad_data if f"{sensor}_mean" in d]
    raw_means = [d[f"{sensor}_mean"] for d in raw_data if f"{sensor}_mean" in d]

    bad_avg = np.mean(bad_means)
    raw_avg = np.mean(raw_means)
    diff = abs(bad_avg - raw_avg)

    print(f"\n{sensor.upper()}:")
    print(f"  Bad 평균: {bad_avg:8.2f}")
    print(f"  Raw 평균: {raw_avg:8.2f}")
    print(f"  차이:     {diff:8.2f}")

# ============================================================================
# 4. 노드 사용 패턴
# ============================================================================
print("\n" + "=" * 100)
print("4. 노드 사용 패턴")
print("=" * 100)

bad_nodes = set()
raw_nodes = set()

for d in bad_data:
    if "start_node" in d:
        bad_nodes.add(d["start_node"])
        bad_nodes.add(d["end_node"])

for d in raw_data:
    if "start_node" in d:
        raw_nodes.add(d["start_node"])
        raw_nodes.add(d["end_node"])

common_nodes = bad_nodes & raw_nodes
bad_only_nodes = bad_nodes - raw_nodes
raw_only_nodes = raw_nodes - bad_nodes

print(f"\nBad 사용 노드: {len(bad_nodes)}개")
print(f"Raw 사용 노드: {len(raw_nodes)}개")
print(f"공통 노드: {len(common_nodes)}개")
print(f"Bad에만: {sorted(bad_only_nodes)}")
print(f"Raw에만: {sorted(raw_only_nodes)}")

if len(common_nodes) / len(bad_nodes | raw_nodes) < 0.3:
    print("\n🎯 결론: **다른 노드 사용**")
    print("  → Bad와 Raw는 다른 지역/경로 측정")

# ============================================================================
# 5. MagX와 위치의 상관관계
# ============================================================================
print("\n" + "=" * 100)
print("5. MagX와 위치의 상관관계")
print("=" * 100)

# Bad + Raw 합쳐서 위치와 MagX 관계 분석
all_data = bad_data + raw_data
all_with_pos = [d for d in all_data if "center_x" in d]

if all_with_pos:
    x_vals = np.array([d["center_x"] for d in all_with_pos])
    magx_vals = np.array([d["magx_mean"] for d in all_with_pos])

    # 상관계수
    corr_x = np.corrcoef(x_vals, magx_vals)[0, 1]

    print(f"\nX 좌표와 MagX 상관계수: {corr_x:.3f}")

    if abs(corr_x) > 0.5:
        print("\n🎯 결론: **위치와 MagX 강한 상관관계**")
        print("  → MagX는 위치에 따라 결정됨")
        print("  → Bad/Raw 차이는 측정 위치 차이")
    elif abs(corr_x) < 0.2:
        print("\n🎯 결론: **위치와 MagX 약한 상관관계**")
        print("  → MagX는 센서/시간에 따라 결정됨")

# ============================================================================
# 최종 종합
# ============================================================================
print("\n" + "=" * 100)
print("🎯 최종 종합 결론")
print("=" * 100)

print("""
분석 항목:
1. 시간: Bad와 Raw의 측정 시간 비교
2. 공간: Bad와 Raw의 측정 위치 비교
3. 센서: 모든 센서 값 비교
4. 노드: 사용한 노드 패턴 비교
5. 상관관계: 위치와 MagX의 관계

위 분석을 종합하여 판단:
- 시간이 다르면 → 측정 세션 차이
- 위치가 다르면 → 경로/지역 차이
- 모든 센서가 다르면 → 센서 캘리브레이션 차이
- MagX만 다르면 → 위치 차이

스크롤업하여 각 분석 결과를 확인하세요.
""")

print("=" * 100)
