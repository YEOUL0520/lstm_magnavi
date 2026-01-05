#!/usr/bin/env python3
"""Bad 데이터 분석 - 왜 bad인지 파악"""
import csv
from pathlib import Path
from collections import defaultdict, Counter

# 데이터 폴더
bad_dir = Path("data/bad")
raw_dir = Path("data/raw")

bad_files = list(bad_dir.glob("*.csv"))
raw_files = list(raw_dir.glob("*.csv"))

print("=" * 80)
print("📊 Bad 데이터 분석")
print("=" * 80)
print(f"Bad 파일: {len(bad_files)}개")
print(f"Raw 파일: {len(raw_files)}개")
print()

# 1. 경로별 분포 비교
print("=" * 80)
print("1. 경로별 분포")
print("=" * 80)

bad_paths = Counter()
raw_paths = Counter()

for f in bad_files:
    parts = f.stem.split("_")
    if len(parts) >= 2:
        path = f"{parts[0]}->{parts[1]}"
        bad_paths[path] += 1

for f in raw_files:
    parts = f.stem.split("_")
    if len(parts) >= 2:
        path = f"{parts[0]}->{parts[1]}"
        raw_paths[path] += 1

print(f"\nBad 데이터 경로 종류: {len(bad_paths)}개")
print(f"Raw 데이터 경로 종류: {len(raw_paths)}개")

# Bad에만 있는 경로
bad_only = set(bad_paths.keys()) - set(raw_paths.keys())
print(f"\n❌ Bad에만 있는 경로: {len(bad_only)}개")
if bad_only:
    print("샘플:")
    for path in sorted(bad_only)[:10]:
        print(f"  - {path}: {bad_paths[path]}개")

# 2. 데이터 길이 분석
print("\n" + "=" * 80)
print("2. 데이터 길이 분석")
print("=" * 80)

bad_lengths = []
raw_lengths = []

for f in bad_files[:50]:  # 샘플 50개
    with f.open() as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        bad_lengths.append(len(rows) - 1)  # 헤더 제외

for f in raw_files[:50]:  # 샘플 50개
    with f.open() as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        raw_lengths.append(len(rows) - 1)

print(f"Bad 평균 길이: {sum(bad_lengths) / len(bad_lengths):.1f} (min={min(bad_lengths)}, max={max(bad_lengths)})")
print(f"Raw 평균 길이: {sum(raw_lengths) / len(raw_lengths):.1f} (min={min(raw_lengths)}, max={max(raw_lengths)})")

# 3. 센서 값 통계 (샘플)
print("\n" + "=" * 80)
print("3. 센서 값 분석 (샘플 10개)")
print("=" * 80)

def analyze_file(file_path):
    """파일의 센서 값 통계"""
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    try:
        magx = [float(row["MagX"]) for row in rows]
        magy = [float(row["MagY"]) for row in rows]
        magz = [float(row["MagZ"]) for row in rows]

        return {
            "file": file_path.name,
            "length": len(rows),
            "magx_mean": sum(magx) / len(magx),
            "magy_mean": sum(magy) / len(magy),
            "magz_mean": sum(magz) / len(magz),
            "magx_std": (sum((x - sum(magx)/len(magx))**2 for x in magx) / len(magx)) ** 0.5,
        }
    except (KeyError, ValueError) as e:
        return {"file": file_path.name, "error": str(e)}

print("\nBad 데이터:")
for f in bad_files[:5]:
    stats = analyze_file(f)
    if stats:
        if "error" in stats:
            print(f"  ❌ {stats['file']}: ERROR - {stats['error']}")
        else:
            print(f"  - {stats['file']}: len={stats['length']}, MagX={stats['magx_mean']:.1f}±{stats['magx_std']:.1f}")

print("\nRaw 데이터:")
for f in raw_files[:5]:
    stats = analyze_file(f)
    if stats:
        if "error" in stats:
            print(f"  ❌ {stats['file']}: ERROR - {stats['error']}")
        else:
            print(f"  - {stats['file']}: len={stats['length']}, MagX={stats['magx_mean']:.1f}±{stats['magx_std']:.1f}")

# 4. 경로 샘플 수 불균형 체크
print("\n" + "=" * 80)
print("4. 경로별 샘플 수 (Raw 데이터)")
print("=" * 80)

print("\n샘플 수가 적은 경로 (5개 이하):")
low_sample_paths = [(path, count) for path, count in raw_paths.items() if count <= 5]
low_sample_paths.sort(key=lambda x: x[1])

for path, count in low_sample_paths[:20]:
    print(f"  - {path}: {count}개")

print(f"\n총 {len(low_sample_paths)}개 경로가 5개 이하")

# 5. 종합 분석
print("\n" + "=" * 80)
print("📈 종합 분석")
print("=" * 80)

print(f"""
1. 파일 수: Bad={len(bad_files)}개, Raw={len(raw_files)}개
2. 경로 종류: Bad={len(bad_paths)}개, Raw={len(raw_paths)}개
3. Bad에만 있는 경로: {len(bad_only)}개
4. 평균 길이: Bad={sum(bad_lengths)/len(bad_lengths):.0f}, Raw={sum(raw_lengths)/len(raw_lengths):.0f}
5. 샘플 부족 경로 (Raw): {len(low_sample_paths)}개

가능한 "Bad" 이유:
  - 특정 경로의 데이터만 모아둔 것
  - 품질 문제 (노이즈, 센서 오류)
  - 길이가 너무 짧거나 긴 데이터
  - 테스트/검증용 데이터
""")

print("=" * 80)
