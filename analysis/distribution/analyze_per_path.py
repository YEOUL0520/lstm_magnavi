#!/usr/bin/env python3
"""경로별 Pitch, Roll, Yaw 변화 분석"""
import csv
import math
from pathlib import Path
import random

data_dir = Path("/Users/yunho/school/lstm/data/raw")
csv_files = list(data_dir.glob("*.csv"))

# 랜덤하게 10개 파일 샘플링
sample_files = random.sample(csv_files, min(10, len(csv_files)))

def stats(values):
    if not values:
        return {}
    mean = sum(values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)),
        "range": max(values) - min(values),
    }

print(f"경로별 Pitch/Roll 변화 분석 ({len(sample_files)}개 경로)\n")
print("=" * 80)
print(f"{'파일명':<25} {'Pitch std':<12} {'Roll std':<12} {'데이터 수':<10}")
print("=" * 80)

pitch_stds = []
roll_stds = []
yaw_stds = []

for csv_file in sample_files:
    with csv_file.open() as f:
        reader = csv.DictReader(f)
        pitch_vals = []
        roll_vals = []
        yaw_vals = []

        for row in reader:
            try:
                pitch_vals.append(float(row["Pitch"]))
                roll_vals.append(float(row["Roll"]))
                yaw_vals.append(float(row["Yaw"]))
            except (ValueError, KeyError):
                continue

    if pitch_vals:
        pitch_stat = stats(pitch_vals)
        roll_stat = stats(roll_vals)
        yaw_stat = stats(yaw_vals)

        pitch_stds.append(pitch_stat['std'])
        roll_stds.append(roll_stat['std'])
        yaw_stds.append(yaw_stat['std'])

        print(f"{csv_file.name:<25} {pitch_stat['std']:>8.3f}°    {roll_stat['std']:>8.3f}°    {len(pitch_vals):>6}개")

print("=" * 80)
print("\n평균 경로 내 변화:")
print(f"  Pitch 평균 std: {sum(pitch_stds) / len(pitch_stds):.3f}° (경로별 평균)")
print(f"  Roll 평균 std:  {sum(roll_stds) / len(roll_stds):.3f}° (경로별 평균)")
print(f"  Yaw 평균 std:   {sum(yaw_stds) / len(yaw_stds):.3f}° (경로별 평균)")

print("\n결론:")
avg_pitch_std = sum(pitch_stds) / len(pitch_stds)
avg_roll_std = sum(roll_stds) / len(roll_stds)

if avg_pitch_std < 3 and avg_roll_std < 3:
    print("  ✅ 한 경로 내에서 Pitch/Roll 변화가 매우 작음 (평균 std < 3°)")
    print("  → 거의 일정한 값 유지, 제거해도 될 가능성 높음")
elif avg_pitch_std < 5 and avg_roll_std < 5:
    print("  🟡 한 경로 내에서 Pitch/Roll 변화가 작음 (평균 std < 5°)")
    print("  → 제거 고려 가능, 실험 필요")
else:
    print("  🔴 한 경로 내에서 Pitch/Roll 변화가 있음 (평균 std >= 5°)")
    print("  → 의미 있는 정보일 가능성, 유지 권장")
