#!/usr/bin/env python3
"""Bad 데이터 심층 분석 - 각 파일의 문제점 파악"""
import csv
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

bad_dir = Path("data/bad")
raw_dir = Path("data/raw")

# 분석할 문제 유형
issues = defaultdict(list)

def analyze_file_deep(file_path, is_bad=True):
    """단일 파일 심층 분석"""
    problems = []

    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return ["EMPTY_FILE"]

    # 1. 길이 체크
    if len(rows) < 100:
        problems.append(f"TOO_SHORT({len(rows)})")
    elif len(rows) > 5000:
        problems.append(f"TOO_LONG({len(rows)})")

    # 2. 센서 값 추출
    try:
        timestamps = [row.get("Timestamp", "") for row in rows]
        magx = [float(row["MagX"]) for row in rows]
        magy = [float(row["MagY"]) for row in rows]
        magz = [float(row["MagZ"]) for row in rows]
        pitch = [float(row["Pitch"]) for row in rows]
        roll = [float(row["Roll"]) for row in rows]
        yaw = [float(row["Yaw"]) for row in rows]
    except (KeyError, ValueError) as e:
        return [f"PARSE_ERROR({e})"]

    # 3. NaN/Inf 체크
    all_values = magx + magy + magz + pitch + roll + yaw
    if any(math.isnan(v) or math.isinf(v) for v in all_values):
        problems.append("NAN_OR_INF")

    # 4. 센서 값 범위 체크
    magx_mean = sum(magx) / len(magx)
    magy_mean = sum(magy) / len(magy)
    magz_mean = sum(magz) / len(magz)

    # 지자기 값이 이상한 경우 (지구 자기장 범위 벗어남)
    if abs(magx_mean) > 100 or abs(magy_mean) > 100 or abs(magz_mean) > 100:
        problems.append(f"MAG_OUT_OF_RANGE(X={magx_mean:.1f},Y={magy_mean:.1f},Z={magz_mean:.1f})")

    # 5. 센서 값 분산 체크 (너무 일정하면 이상)
    magx_std = np.std(magx)
    magy_std = np.std(magy)
    magz_std = np.std(magz)

    if magx_std < 0.1 and magy_std < 0.1 and magz_std < 0.1:
        problems.append("SENSOR_FROZEN")

    # 6. 급격한 점프 체크
    magx_jumps = sum(1 for i in range(1, len(magx)) if abs(magx[i] - magx[i-1]) > 50)
    if magx_jumps > len(magx) * 0.1:  # 10% 이상 점프
        problems.append(f"EXCESSIVE_JUMPS({magx_jumps})")

    # 7. 버튼 정보 체크
    if "Highlighted" in rows[0] and "RightAngle" in rows[0]:
        highlighted = [row.get("Highlighted", "false") for row in rows]
        right_angle = [row.get("RightAngle", "false") for row in rows]

        has_button = any(h.lower() == "true" for h in highlighted + right_angle)
        if not has_button:
            problems.append("NO_BUTTON_PRESS")

    # 8. 타임스탬프 순서 체크
    if timestamps and len(timestamps) > 1:
        # 타임스탬프가 역순이거나 중복되는지
        try:
            ts_valid = [t for t in timestamps if t]
            if len(ts_valid) >= 2:
                # 첫 번째와 마지막 비교
                if ts_valid[0] > ts_valid[-1]:
                    problems.append("REVERSE_TIMESTAMP")
        except:
            pass

    # 9. 센서 캘리브레이션 차이 (Bad vs Raw 기준)
    # Raw 데이터의 MagX 평균은 약 -20~-30
    # Bad 데이터의 MagX 평균은 약 30~40
    if is_bad:
        if magx_mean < 0:  # Bad인데 Raw처럼 음수
            problems.append(f"CALIBRATION_MISMATCH(MagX={magx_mean:.1f})")
    else:
        if magx_mean > 0:  # Raw인데 Bad처럼 양수
            problems.append(f"CALIBRATION_MISMATCH(MagX={magx_mean:.1f})")

    # 10. 데이터 통계
    stats = {
        "length": len(rows),
        "magx_mean": magx_mean,
        "magx_std": magx_std,
        "magy_mean": magy_mean,
        "magz_mean": magz_mean,
    }

    return problems, stats

print("=" * 100)
print("🔍 Bad 데이터 심층 분석")
print("=" * 100)
print()

# Bad 데이터 전체 분석
print("📊 Bad 데이터 분석 중...")
bad_results = {}
for f in bad_dir.glob("*.csv"):
    result = analyze_file_deep(f, is_bad=True)
    if isinstance(result, tuple):
        problems, stats = result
        bad_results[f.name] = {"problems": problems, "stats": stats}
    else:
        bad_results[f.name] = {"problems": result, "stats": None}

# Raw 데이터 샘플 분석 (비교용)
print("📊 Raw 데이터 분석 중 (샘플)...")
raw_results = {}
for f in list(raw_dir.glob("*.csv"))[:50]:
    result = analyze_file_deep(f, is_bad=False)
    if isinstance(result, tuple):
        problems, stats = result
        raw_results[f.name] = {"problems": problems, "stats": stats}
    else:
        raw_results[f.name] = {"problems": result, "stats": None}

print()
print("=" * 100)
print("📈 분석 결과 요약")
print("=" * 100)

# 문제 유형별 분류
issue_types = defaultdict(list)
for fname, data in bad_results.items():
    if data["problems"]:
        for problem in data["problems"]:
            issue_type = problem.split("(")[0]  # 괄호 앞부분만
            issue_types[issue_type].append(fname)
    else:
        issue_types["NO_ISSUE"].append(fname)

print(f"\n총 Bad 파일: {len(bad_results)}개")
print(f"\n문제 유형별 분류:")
print("-" * 100)

for issue_type, files in sorted(issue_types.items(), key=lambda x: len(x[1]), reverse=True):
    count = len(files)
    percentage = (count / len(bad_results)) * 100
    print(f"\n{issue_type}: {count}개 ({percentage:.1f}%)")

    # 샘플 5개만 출력
    for fname in files[:5]:
        full_problems = bad_results[fname]["problems"]
        print(f"  - {fname}: {', '.join(full_problems)}")

    if len(files) > 5:
        print(f"  ... 외 {len(files) - 5}개")

# 통계 비교
print("\n" + "=" * 100)
print("📊 센서 값 통계 비교")
print("=" * 100)

bad_stats = [d["stats"] for d in bad_results.values() if d["stats"]]
raw_stats = [d["stats"] for d in raw_results.values() if d["stats"]]

if bad_stats and raw_stats:
    bad_magx_mean = sum(s["magx_mean"] for s in bad_stats) / len(bad_stats)
    raw_magx_mean = sum(s["magx_mean"] for s in raw_stats) / len(raw_stats)

    bad_magx_std_avg = sum(s["magx_std"] for s in bad_stats) / len(bad_stats)
    raw_magx_std_avg = sum(s["magx_std"] for s in raw_stats) / len(raw_stats)

    print(f"\nMagX 평균:")
    print(f"  Bad: {bad_magx_mean:.2f}μT")
    print(f"  Raw: {raw_magx_mean:.2f}μT")
    print(f"  차이: {abs(bad_magx_mean - raw_magx_mean):.2f}μT")

    print(f"\nMagX 표준편차 (평균):")
    print(f"  Bad: {bad_magx_std_avg:.2f}")
    print(f"  Raw: {raw_magx_std_avg:.2f}")

# 구체적 문제 파일 출력
print("\n" + "=" * 100)
print("🚨 심각한 문제가 있는 파일들")
print("=" * 100)

serious_problems = ["PARSE_ERROR", "NAN_OR_INF", "SENSOR_FROZEN", "EXCESSIVE_JUMPS"]
serious_files = []

for fname, data in bad_results.items():
    for problem in data["problems"]:
        if any(sp in problem for sp in serious_problems):
            serious_files.append((fname, data["problems"]))
            break

if serious_files:
    print(f"\n총 {len(serious_files)}개 파일에 심각한 문제 발견:")
    for fname, problems in serious_files[:20]:
        print(f"  ❌ {fname}: {', '.join(problems)}")
else:
    print("\n✅ 심각한 문제가 있는 파일 없음")

# 결론
print("\n" + "=" * 100)
print("🎯 결론")
print("=" * 100)

print(f"""
1. **센서 캘리브레이션 차이**
   - Bad 데이터의 MagX 평균: {bad_magx_mean:.1f}μT
   - Raw 데이터의 MagX 평균: {raw_magx_mean:.1f}μT
   - 약 {abs(bad_magx_mean - raw_magx_mean):.1f}μT 차이 → 다른 측정 세션

2. **주요 문제**
""")

for issue_type in sorted(issue_types.keys(), key=lambda x: len(issue_types[x]), reverse=True)[:5]:
    count = len(issue_types[issue_type])
    print(f"   - {issue_type}: {count}개 ({count/len(bad_results)*100:.1f}%)")

print(f"""
3. **권장 사항**
   - Bad 데이터는 Raw와 센서 기준점이 다름
   - 분리해서 사용하거나, 정규화 방식 통일 필요
   - 심각한 문제 파일: {len(serious_files)}개 (제외 권장)
""")

print("=" * 100)
