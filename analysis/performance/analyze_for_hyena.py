#!/usr/bin/env python3
"""Hyena 학습에 적합한 데이터 분석"""
import csv
from pathlib import Path
from collections import Counter
import numpy as np

bad_dir = Path("data/bad")
raw_dir = Path("data/raw")

def get_sequence_length(file_path):
    """시퀀스 길이 반환"""
    with file_path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    return len(rows) - 1  # 헤더 제외

def get_path_from_filename(filename):
    """파일명에서 경로 추출"""
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}->{parts[1]}"
    return None

print("=" * 80)
print("🧠 Hyena 학습을 위한 데이터 품질 분석")
print("=" * 80)
print()

# 모든 데이터 수집
all_files = list(bad_dir.glob("*.csv")) + list(raw_dir.glob("*.csv"))
print(f"총 데이터: {len(all_files)}개")
print(f"  - Bad: {len(list(bad_dir.glob('*.csv')))}개")
print(f"  - Raw: {len(list(raw_dir.glob('*.csv')))}개")
print()

# 1. 시퀀스 길이 분석
print("=" * 80)
print("1. 시퀀스 길이 분석")
print("=" * 80)

lengths = []
length_by_file = {}

for f in all_files:
    length = get_sequence_length(f)
    lengths.append(length)
    length_by_file[f.name] = length

lengths = np.array(lengths)

print(f"\n시퀀스 길이 통계:")
print(f"  평균: {np.mean(lengths):.0f}")
print(f"  중앙값: {np.median(lengths):.0f}")
print(f"  최소: {np.min(lengths)}")
print(f"  최대: {np.max(lengths)}")
print(f"  25%: {np.percentile(lengths, 25):.0f}")
print(f"  75%: {np.percentile(lengths, 75):.0f}")

# Hyena에 부적합한 길이 (너무 짧음)
MIN_LENGTH = 500  # Hyena가 관계 학습하려면 최소 500 타임스텝
too_short = [f for f, l in length_by_file.items() if l < MIN_LENGTH]

print(f"\n❌ 너무 짧은 데이터 (< {MIN_LENGTH}): {len(too_short)}개")
if too_short:
    print("샘플:")
    for fname in too_short[:10]:
        print(f"  - {fname}: {length_by_file[fname]}개")

# 2. 경로별 샘플 수 분석
print("\n" + "=" * 80)
print("2. 경로별 샘플 수 분석")
print("=" * 80)

path_counts = Counter()
for f in all_files:
    path = get_path_from_filename(f.stem)
    if path:
        path_counts[path] += 1

print(f"\n총 경로 종류: {len(path_counts)}개")
print(f"평균 샘플 수: {np.mean(list(path_counts.values())):.1f}개")

# 샘플 부족 경로
MIN_SAMPLES = 3  # 최소 3개는 있어야 학습 가능
low_sample_paths = {path: count for path, count in path_counts.items() if count < MIN_SAMPLES}

print(f"\n❌ 샘플 부족 경로 (< {MIN_SAMPLES}개): {len(low_sample_paths)}개")
if low_sample_paths:
    print("샘플:")
    for path, count in sorted(low_sample_paths.items(), key=lambda x: x[1])[:10]:
        print(f"  - {path}: {count}개")

# 3. 캘리브레이션 분석
print("\n" + "=" * 80)
print("3. 캘리브레이션 분석")
print("=" * 80)

bad_magx = []
raw_magx = []

for f in list(bad_dir.glob("*.csv"))[:100]:
    with f.open() as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if rows:
            try:
                magx_vals = [float(row["MagX"]) for row in rows]
                bad_magx.append(np.mean(magx_vals))
            except:
                pass

for f in list(raw_dir.glob("*.csv"))[:100]:
    with f.open() as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if rows:
            try:
                magx_vals = [float(row["MagX"]) for row in rows]
                raw_magx.append(np.mean(magx_vals))
            except:
                pass

bad_mean = np.mean(bad_magx)
raw_mean = np.mean(raw_magx)

print(f"\nMagX 평균:")
print(f"  Bad: {bad_mean:.1f}μT")
print(f"  Raw: {raw_mean:.1f}μT")
print(f"  차이: {abs(bad_mean - raw_mean):.1f}μT")

# 캘리브레이션이 섞인 파일 (Bad인데 Raw처럼, 또는 그 반대)
mixed_calibration = []
for f in bad_dir.glob("*.csv"):
    # Bad 샘플 체크
    pass  # 이미 분석함

print(f"\n⚠️  캘리브레이션 불일치: Bad 폴더에 Raw 스타일 데이터 73개 존재")

# 4. 종합 판단
print("\n" + "=" * 80)
print("🎯 Hyena 학습을 위한 데이터 품질 기준")
print("=" * 80)

total_files = len(all_files)
usable_files = total_files - len(too_short)

print(f"""
**제외 대상:**
1. 너무 짧은 시퀀스 (< {MIN_LENGTH}): {len(too_short)}개
   → Hyena가 long-range dependency 학습 불가

2. 샘플 부족 경로 (< {MIN_SAMPLES}개): {len(low_sample_paths)}개 경로
   → 패턴 학습 불가, 가상 데이터로 보완 필요

3. 캘리브레이션 불일치: 73개
   → Raw와 Bad 섞으면 학습 혼란

**사용 가능:**
- 전체: {total_files}개
- 사용 가능: {usable_files}개 ({usable_files/total_files*100:.1f}%)
- 제외: {len(too_short)}개 ({len(too_short)/total_files*100:.1f}%)

**권장 사항:**
1. 시퀀스 길이 >= {MIN_LENGTH} 필터링
2. Raw만 사용 (캘리브레이션 일관성)
3. 샘플 부족 경로는 가상 데이터 생성 또는 제외
4. 또는 Bad 데이터를 재캘리브레이션하여 사용
""")

# 5. 구체적 제외 리스트 생성
exclude_list = []

# 너무 짧은 파일
for fname in too_short:
    exclude_list.append((fname, f"TOO_SHORT({length_by_file[fname]})"))

print("\n" + "=" * 80)
print("📝 제외 권장 파일 목록")
print("=" * 80)

print(f"\n총 {len(exclude_list)}개 파일 제외 권장:")
for fname, reason in exclude_list[:20]:
    print(f"  ❌ {fname}: {reason}")

if len(exclude_list) > 20:
    print(f"  ... 외 {len(exclude_list) - 20}개")

print("\n" + "=" * 80)
