#!/usr/bin/env python3
"""캘리브레이션 차이 원인 분석: 경로 vs 센서"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

bad_dir = Path("data/bad")
raw_dir = Path("data/raw")

def get_magx_mean(file_path):
    """파일의 MagX 평균 계산"""
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    try:
        magx_vals = [float(row["MagX"]) for row in rows]
        return np.mean(magx_vals)
    except (KeyError, ValueError):
        return None

def get_path_from_filename(filename):
    """파일명에서 경로 추출"""
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}->{parts[1]}"
    return None

print("=" * 80)
print("🔍 캘리브레이션 차이 원인 분석")
print("=" * 80)
print()

# 1. 경로별 MagX 평균 수집
print("📊 데이터 수집 중...")

bad_by_path = defaultdict(list)
raw_by_path = defaultdict(list)

for f in bad_dir.glob("*.csv"):
    path = get_path_from_filename(f.stem)
    magx_mean = get_magx_mean(f)
    if path and magx_mean is not None:
        bad_by_path[path].append((f.name, magx_mean))

for f in raw_dir.glob("*.csv"):
    path = get_path_from_filename(f.stem)
    magx_mean = get_magx_mean(f)
    if path and magx_mean is not None:
        raw_by_path[path].append((f.name, magx_mean))

print(f"Bad 경로: {len(bad_by_path)}개")
print(f"Raw 경로: {len(raw_by_path)}개")
print()

# 2. 같은 경로가 둘 다 있는지 확인
print("=" * 80)
print("1. 같은 경로 비교 (경로 차이 vs 센서 차이)")
print("=" * 80)

common_paths = set(bad_by_path.keys()) & set(raw_by_path.keys())
print(f"\n같은 경로가 Bad와 Raw에 모두 존재: {len(common_paths)}개")

if common_paths:
    print("\n같은 경로의 MagX 평균 비교:")
    print("-" * 80)

    differences = []
    for path in sorted(common_paths)[:10]:
        bad_means = [m for _, m in bad_by_path[path]]
        raw_means = [m for _, m in raw_by_path[path]]

        bad_avg = np.mean(bad_means)
        raw_avg = np.mean(raw_means)
        diff = bad_avg - raw_avg
        differences.append(diff)

        print(f"{path:10s}: Bad={bad_avg:6.1f}μT, Raw={raw_avg:6.1f}μT, 차이={diff:+6.1f}μT")

    avg_diff = np.mean(differences)
    print(f"\n평균 차이: {avg_diff:+.1f}μT")

    if abs(avg_diff) > 30:
        print("\n🎯 결론: **센서/측정 세션 차이**")
        print("  → 같은 경로인데도 40μT 가까이 차이남")
        print("  → 경로가 아니라 측정 시기/센서가 다름")
    else:
        print("\n🎯 결론: **경로 차이**")
        print("  → 같은 경로는 비슷한 값")
        print("  → Bad와 Raw가 서로 다른 경로 위주")

else:
    print("\n⚠️  Bad와 Raw에 공통 경로 없음")
    print("  → 완전히 다른 경로들로 구성")

# 3. 경로 내 분산 vs 전체 분산
print("\n" + "=" * 80)
print("2. 경로 내 분산 vs 전체 분산")
print("=" * 80)

# Bad 데이터
bad_all_means = []
bad_within_path_var = []

for path, files in bad_by_path.items():
    means = [m for _, m in files]
    bad_all_means.extend(means)
    if len(means) > 1:
        bad_within_path_var.append(np.var(means))

bad_total_var = np.var(bad_all_means)
bad_within_var = np.mean(bad_within_path_var) if bad_within_path_var else 0

print(f"\nBad 데이터:")
print(f"  전체 분산: {bad_total_var:.1f}")
print(f"  경로 내 평균 분산: {bad_within_var:.1f}")
print(f"  비율: {bad_within_var / bad_total_var * 100:.1f}%")

# Raw 데이터
raw_all_means = []
raw_within_path_var = []

for path, files in raw_by_path.items():
    means = [m for _, m in files]
    raw_all_means.extend(means)
    if len(means) > 1:
        raw_within_path_var.append(np.var(means))

raw_total_var = np.var(raw_all_means)
raw_within_var = np.mean(raw_within_path_var) if raw_within_path_var else 0

print(f"\nRaw 데이터:")
print(f"  전체 분산: {raw_total_var:.1f}")
print(f"  경로 내 평균 분산: {raw_within_var:.1f}")
print(f"  비율: {raw_within_var / raw_total_var * 100:.1f}%")

if bad_within_var / bad_total_var < 0.3 and raw_within_var / raw_total_var < 0.3:
    print("\n🎯 경로 내 분산이 작음 (< 30%)")
    print("  → 같은 경로는 비슷한 값")
    print("  → MagX 변화는 **경로에 따라 결정**됨")

# 4. Bad와 Raw의 경로 중복도
print("\n" + "=" * 80)
print("3. 경로 중복 분석")
print("=" * 80)

bad_only = set(bad_by_path.keys()) - set(raw_by_path.keys())
raw_only = set(raw_by_path.keys()) - set(bad_by_path.keys())

print(f"\nBad에만 있는 경로: {len(bad_only)}개")
print(f"Raw에만 있는 경로: {len(raw_only)}개")
print(f"공통 경로: {len(common_paths)}개")

# 5. 경로별 MagX 평균 분포
print("\n" + "=" * 80)
print("4. 각 데이터셋 내부 경로별 MagX 범위")
print("=" * 80)

bad_path_means = {path: np.mean([m for _, m in files]) for path, files in bad_by_path.items()}
raw_path_means = {path: np.mean([m for _, m in files]) for path, files in raw_by_path.items()}

print(f"\nBad 데이터 경로별 MagX 범위:")
print(f"  최소: {min(bad_path_means.values()):.1f}μT")
print(f"  최대: {max(bad_path_means.values()):.1f}μT")
print(f"  범위: {max(bad_path_means.values()) - min(bad_path_means.values()):.1f}μT")

print(f"\nRaw 데이터 경로별 MagX 범위:")
print(f"  최소: {min(raw_path_means.values()):.1f}μT")
print(f"  최대: {max(raw_path_means.values()):.1f}μT")
print(f"  범위: {max(raw_path_means.values()) - min(raw_path_means.values()):.1f}μT")

bad_range = max(bad_path_means.values()) - min(bad_path_means.values())
raw_range = max(raw_path_means.values()) - min(raw_path_means.values())

if bad_range < 30 and raw_range < 30:
    print("\n🎯 각 데이터셋 내부에서 경로별 차이 작음 (< 30μT)")
    print("  → Bad 내부는 비슷, Raw 내부는 비슷")
    print("  → 하지만 Bad와 Raw 사이는 40μT 차이")
    print("  → **센서/측정 세션 차이가 원인**")

# 최종 결론
print("\n" + "=" * 80)
print("🎯 최종 결론")
print("=" * 80)

print(f"""
1. 공통 경로: {len(common_paths)}개
   → Bad와 Raw가 {len(common_paths)}개 경로를 공유

2. Bad/Raw 각각 내부 범위:
   - Bad: {bad_range:.1f}μT
   - Raw: {raw_range:.1f}μT

3. Bad vs Raw 평균 차이: 40.3μT
""")

if len(common_paths) > 5:
    print("✅ **센서/측정 세션 차이가 주 원인**")
    print("   - 같은 경로도 40μT 차이남")
    print("   - 다른 날짜/시간/센서로 측정")
    print("   - 캘리브레이션 오프셋 차이")
else:
    print("⚠️  **경로 차이 + 센서 차이 복합**")
    print("   - 공통 경로가 거의 없음")
    print("   - Bad와 Raw가 다른 경로 위주")
    print("   - 하지만 40μT는 경로만으로 설명 어려움")

print("\n권장:")
print("  → 센서 차이가 주 원인이므로")
print("  → Adaptive Normalization 또는 별도 BASE_MAG 사용")
print("=" * 80)
