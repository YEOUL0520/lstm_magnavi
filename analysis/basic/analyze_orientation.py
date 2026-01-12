#!/usr/bin/env python3
"""Pitch, Roll, Yaw 통계 분석"""
import csv
import math
from pathlib import Path
import random

data_dir = Path("/Users/yunho/school/lstm/data/raw")
csv_files = list(data_dir.glob("*.csv"))

# 랜덤하게 20개 파일 샘플링
sample_files = random.sample(csv_files, min(20, len(csv_files)))

all_pitch = []
all_roll = []
all_yaw = []

for csv_file in sample_files:
    with csv_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pitch = float(row["Pitch"])
                roll = float(row["Roll"])
                yaw = float(row["Yaw"])
                all_pitch.append(pitch)
                all_roll.append(roll)
                all_yaw.append(yaw)
            except (ValueError, KeyError):
                continue

def stats(values):
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "std": math.sqrt(sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)),
        "range": max(values) - min(values),
    }

print(f"샘플 파일: {len(sample_files)}개")
print(f"총 데이터 포인트: {len(all_pitch):,}개\n")

print("=" * 60)
print("Pitch 통계:")
pitch_stats = stats(all_pitch)
for k, v in pitch_stats.items():
    print(f"  {k:8s}: {v:8.3f}°")

print("\n" + "=" * 60)
print("Roll 통계:")
roll_stats = stats(all_roll)
for k, v in roll_stats.items():
    print(f"  {k:8s}: {v:8.3f}°")

print("\n" + "=" * 60)
print("Yaw 통계:")
yaw_stats = stats(all_yaw)
for k, v in yaw_stats.items():
    print(f"  {k:8s}: {v:8.3f}°")

print("\n" + "=" * 60)
print("분석 결과:")
print(f"  Pitch 변화폭: {pitch_stats['range']:.3f}° (std={pitch_stats['std']:.3f})")
print(f"  Roll 변화폭:  {roll_stats['range']:.3f}° (std={roll_stats['std']:.3f})")
print(f"  Yaw 변화폭:   {yaw_stats['range']:.3f}° (std={yaw_stats['std']:.3f})")

print("\n결론:")
if pitch_stats['std'] < 5 and roll_stats['std'] < 5:
    print("  ✅ Pitch/Roll 변화가 매우 작음 (std < 5°)")
    print("  → 제거해도 될 가능성 높음")
else:
    print("  ⚠️  Pitch/Roll 변화가 있음")
    print("  → 추가 분석 필요")
