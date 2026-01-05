#!/usr/bin/env python3
"""좋은 Bad 파일들을 Raw 폴더로 이동 (필요시 캘리브레이션 보정)
Note: 루트 디렉토리에서 실행하세요: python analysis/move_good_bad_to_raw.py
"""
import csv
import shutil
from pathlib import Path
import numpy as np

bad_dir = Path("data/bad")
raw_dir = Path("data/raw")
good_bad_file_list = Path("analysis/outputs/good_bad_files.txt")

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

def apply_calibration_offset(input_path, output_path, offset):
    """MagX에 offset 적용하여 새 파일 생성"""
    with input_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # MagX에 offset 적용
    for row in rows:
        try:
            original_magx = float(row["MagX"])
            row["MagX"] = str(original_magx + offset)
        except (KeyError, ValueError):
            pass

    # 새 파일로 저장
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

print("=" * 80)
print("📦 Bad 폴더의 좋은 파일들을 Raw 폴더로 이동")
print("=" * 80)
print()

# 좋은 bad 파일 리스트 읽기
with good_bad_file_list.open() as f:
    good_files = [line.strip() for line in f if line.strip() and not line.startswith("#")]

print(f"처리할 파일: {len(good_files)}개")
print()

# 통계
copied_directly = []
calibration_corrected = []
failed = []

# 각 파일 처리
for filename in good_files:
    src_path = bad_dir / filename

    if not src_path.exists():
        print(f"❌ {filename}: 파일 없음")
        failed.append(filename)
        continue

    # MagX 평균 확인
    magx_mean = get_magx_mean(src_path)

    if magx_mean is None:
        print(f"❌ {filename}: MagX 읽기 실패")
        failed.append(filename)
        continue

    dst_path = raw_dir / filename

    # Raw 스타일 (MagX < 0) → 그냥 복사
    if magx_mean < 0:
        shutil.copy2(src_path, dst_path)
        copied_directly.append((filename, magx_mean))
        print(f"✅ {filename}: 직접 복사 (MagX={magx_mean:.1f}μT)")

    # Bad 스타일 (MagX > 0) → 캘리브레이션 보정 후 복사
    else:
        # Bad 평균(+19.9) → Raw 평균(-20.4) 변환
        # 약 -40μT 오프셋 필요
        offset = -40.3
        apply_calibration_offset(src_path, dst_path, offset)
        calibration_corrected.append((filename, magx_mean, magx_mean + offset))
        print(f"🔧 {filename}: 보정 후 복사 (MagX={magx_mean:.1f}μT → {magx_mean + offset:.1f}μT)")

print()
print("=" * 80)
print("📊 처리 결과")
print("=" * 80)

print(f"\n✅ 직접 복사 (Raw 스타일): {len(copied_directly)}개")
if copied_directly:
    print("샘플:")
    for fname, magx in copied_directly[:10]:
        print(f"  - {fname}: MagX={magx:.1f}μT")
    if len(copied_directly) > 10:
        print(f"  ... 외 {len(copied_directly) - 10}개")

print(f"\n🔧 캘리브레이션 보정: {len(calibration_corrected)}개")
if calibration_corrected:
    print("샘플 (Before → After):")
    for fname, before, after in calibration_corrected[:10]:
        print(f"  - {fname}: {before:.1f}μT → {after:.1f}μT")
    if len(calibration_corrected) > 10:
        print(f"  ... 외 {len(calibration_corrected) - 10}개")

if failed:
    print(f"\n❌ 실패: {len(failed)}개")
    for fname in failed:
        print(f"  - {fname}")

print(f"\n총계:")
print(f"  성공: {len(copied_directly) + len(calibration_corrected)}개")
print(f"  실패: {len(failed)}개")
print(f"  Raw 폴더 총 파일: {len(list(raw_dir.glob('*.csv')))}개")

print()
print("=" * 80)
