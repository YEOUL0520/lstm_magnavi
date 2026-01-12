#!/usr/bin/env python3
"""파일별 품질 분석 - 오차 예측 및 좋은 파일 선별
Note: 루트 디렉토리에서 실행하세요: python analysis/analyze_file_quality.py
"""
import csv
from pathlib import Path
from collections import Counter
import numpy as np

bad_dir = Path("data/bad")
raw_dir = Path("data/raw")

def analyze_quality(file_path):
    """파일 품질 점수 계산"""
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 100:
        return None

    try:
        magx = np.array([float(row["MagX"]) for row in rows])
        magy = np.array([float(row["MagY"]) for row in rows])
        magz = np.array([float(row["MagZ"]) for row in rows])

        quality = {
            "filename": file_path.name,
            "folder": file_path.parent.name,
            "length": len(rows),
            "magx_mean": np.mean(magx),
            "magx_std": np.std(magx),
            "magy_std": np.std(magy),
            "magz_std": np.std(magz),
        }

        # 경로 정보
        parts = file_path.stem.split("_")
        if len(parts) >= 2:
            quality["path"] = f"{parts[0]}->{parts[1]}"
            quality["start"] = int(parts[0])
            quality["end"] = int(parts[1])

        # 품질 점수 계산
        score = 0

        # 1. 길이 점수 (500 이상이면 좋음)
        if quality["length"] >= 1000:
            score += 3
        elif quality["length"] >= 500:
            score += 2
        elif quality["length"] >= 250:
            score += 1

        # 2. 센서 안정성 (std가 너무 작거나 크면 나쁨)
        if 5 < quality["magx_std"] < 20:
            score += 2
        elif 3 < quality["magx_std"] < 30:
            score += 1

        # 3. 노이즈 체크 (급격한 점프)
        jumps = np.sum(np.abs(np.diff(magx)) > 30)
        if jumps < len(magx) * 0.01:  # 1% 미만
            score += 2
        elif jumps < len(magx) * 0.05:  # 5% 미만
            score += 1

        quality["score"] = score
        quality["jumps"] = jumps

        return quality

    except Exception as e:
        return None

print("=" * 100)
print("📊 파일별 품질 분석")
print("=" * 100)
print()

# 전체 파일 분석
print("분석 중...")
all_files = []

for f in bad_dir.glob("*.csv"):
    q = analyze_quality(f)
    if q:
        all_files.append(q)

for f in raw_dir.glob("*.csv"):
    q = analyze_quality(f)
    if q:
        all_files.append(q)

print(f"총 {len(all_files)}개 파일 분석 완료")
print()

# ============================================================================
# 1. 품질 점수 분포
# ============================================================================
print("=" * 100)
print("1. 품질 점수 분포")
print("=" * 100)

bad_files = [f for f in all_files if f["folder"] == "bad"]
raw_files = [f for f in all_files if f["folder"] == "raw"]

bad_scores = [f["score"] for f in bad_files]
raw_scores = [f["score"] for f in raw_files]

print(f"\nBad 폴더 평균 점수: {np.mean(bad_scores):.2f}")
print(f"Raw 폴더 평균 점수: {np.mean(raw_scores):.2f}")

# ============================================================================
# 2. 문제 파일 (낮은 점수)
# ============================================================================
print("\n" + "=" * 100)
print("2. 문제 파일 (품질 점수 낮음)")
print("=" * 100)

# 점수 3 이하
low_quality = sorted([f for f in all_files if f["score"] <= 3],
                     key=lambda x: x["score"])

print(f"\n품질 점수 ≤ 3: {len(low_quality)}개")
print("\n문제 파일 샘플 (점수 낮은 순):")
print("-" * 100)
print(f"{'파일명':<30} {'폴더':<8} {'점수':<5} {'길이':<8} {'MagX std':<10} {'점프':<8} {'경로':<10}")
print("-" * 100)

for f in low_quality[:30]:
    print(f"{f['filename']:<30} {f['folder']:<8} {f['score']:<5} "
          f"{f['length']:<8} {f['magx_std']:<10.2f} {f['jumps']:<8} {f.get('path', 'N/A'):<10}")

# 문제 파일의 공통점
print("\n문제 파일 특징:")
lengths = [f["length"] for f in low_quality]
stds = [f["magx_std"] for f in low_quality]
print(f"  평균 길이: {np.mean(lengths):.0f} (너무 짧음)")
print(f"  평균 std: {np.mean(stds):.2f} (불안정)")

# ============================================================================
# 3. Bad 폴더에서 좋은 파일
# ============================================================================
print("\n" + "=" * 100)
print("3. Bad 폴더에서 품질 좋은 파일")
print("=" * 100)

# 점수 5 이상
good_bad_files = sorted([f for f in bad_files if f["score"] >= 5],
                        key=lambda x: x["score"], reverse=True)

print(f"\nBad 폴더 중 품질 좋은 파일 (점수 ≥ 5): {len(good_bad_files)}개")
print("\n좋은 파일 리스트:")
print("-" * 100)
print(f"{'파일명':<30} {'점수':<5} {'길이':<8} {'MagX 평균':<12} {'경로':<10}")
print("-" * 100)

for f in good_bad_files[:50]:
    print(f"{f['filename']:<30} {f['score']:<5} {f['length']:<8} "
          f"{f['magx_mean']:<12.2f} {f.get('path', 'N/A'):<10}")

if len(good_bad_files) > 50:
    print(f"... 외 {len(good_bad_files) - 50}개")

# ============================================================================
# 4. Raw 스타일 캘리브레이션을 가진 Bad 파일
# ============================================================================
print("\n" + "=" * 100)
print("4. Bad 폴더 중 Raw 스타일 캘리브레이션 (바로 사용 가능)")
print("=" * 100)

# MagX 평균이 0 이하 (Raw 스타일)
raw_style_bad = [f for f in bad_files if f["magx_mean"] < 0 and f["score"] >= 4]
raw_style_bad = sorted(raw_style_bad, key=lambda x: x["score"], reverse=True)

print(f"\nBad 폴더 중 Raw 스타일 (MagX < 0, 점수 ≥ 4): {len(raw_style_bad)}개")
print("이 파일들은 현재 BASE_MAG으로 바로 사용 가능:")
print("-" * 100)

for f in raw_style_bad[:30]:
    print(f"{f['filename']:<30} 점수={f['score']}, 길이={f['length']}, "
          f"MagX={f['magx_mean']:.2f}μT")

if len(raw_style_bad) > 30:
    print(f"... 외 {len(raw_style_bad) - 30}개")

# ============================================================================
# 5. 경로별 품질
# ============================================================================
print("\n" + "=" * 100)
print("5. 경로별 품질 분석")
print("=" * 100)

path_quality = {}
for f in all_files:
    if "path" in f:
        path = f["path"]
        if path not in path_quality:
            path_quality[path] = []
        path_quality[path].append(f["score"])

# 평균 점수가 낮은 경로
bad_paths = [(p, np.mean(scores)) for p, scores in path_quality.items() if np.mean(scores) < 4]
bad_paths = sorted(bad_paths, key=lambda x: x[1])

print(f"\n품질 낮은 경로 (평균 점수 < 4): {len(bad_paths)}개")
for path, avg_score in bad_paths[:20]:
    print(f"  {path:10s}: 평균 점수 {avg_score:.2f}")

# ============================================================================
# 6. 최종 권장 사항
# ============================================================================
print("\n" + "=" * 100)
print("🎯 최종 권장 사항")
print("=" * 100)

print(f"""
**사용 권장 파일:**
1. Raw 폴더 전체: {len(raw_files)}개
2. Bad 폴더 중 품질 좋음 (점수 ≥ 5): {len(good_bad_files)}개
3. Bad 폴더 중 Raw 스타일 (바로 사용 가능): {len(raw_style_bad)}개

**제외 권장 파일:**
- 품질 점수 ≤ 3: {len(low_quality)}개
  → 너무 짧거나 불안정

**데이터 증가:**
- 현재 (Raw만): {len(raw_files)}개
- 추가 가능 (Bad 품질 좋음): +{len(good_bad_files)}개
- 합계: {len(raw_files) + len(good_bad_files)}개 ({(len(raw_files) + len(good_bad_files)) / len(raw_files) * 100:.0f}%)

**방법:**
1. Raw 스타일 Bad 파일 → 현재 BASE_MAG으로 바로 사용
2. 나머지 좋은 Bad 파일 → Adaptive Normalization 필요
""")

# ============================================================================
# 7. 구체적 파일 리스트 저장
# ============================================================================
print("\n" + "=" * 100)
print("📝 파일 리스트 저장")
print("=" * 100)

# 좋은 bad 파일 리스트
with open("analysis/outputs/good_bad_files.txt", "w") as f:
    f.write("# Bad 폴더 중 품질 좋은 파일 (점수 >= 5)\n")
    for file in good_bad_files:
        f.write(f"{file['filename']}\n")

print(f"✅ analysis/outputs/good_bad_files.txt 저장 완료 ({len(good_bad_files)}개)")

# Raw 스타일 bad 파일 리스트
with open("analysis/outputs/raw_style_bad_files.txt", "w") as f:
    f.write("# Bad 폴더 중 Raw 스타일 캘리브레이션 (바로 사용 가능)\n")
    for file in raw_style_bad:
        f.write(f"{file['filename']}\n")

print(f"✅ analysis/outputs/raw_style_bad_files.txt 저장 완료 ({len(raw_style_bad)}개)")

# 제외할 파일 리스트
with open("analysis/outputs/exclude_files.txt", "w") as f:
    f.write("# 품질 낮은 파일 (점수 <= 3)\n")
    for file in low_quality:
        f.write(f"{file['filename']}\n")

print(f"✅ analysis/outputs/exclude_files.txt 저장 완료 ({len(low_quality)}개)")

print("\n" + "=" * 100)
