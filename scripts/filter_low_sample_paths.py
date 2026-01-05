#!/usr/bin/env python3
"""
샘플 수가 적은 경로를 제거하여 균형잡힌 데이터셋 생성
"""
from pathlib import Path
from collections import defaultdict
import shutil

def main():
    raw_dir = Path('data/raw')
    filtered_dir = Path('data/raw_filtered')
    low_sample_dir = Path('data/low_sample')

    filtered_dir.mkdir(exist_ok=True)
    low_sample_dir.mkdir(exist_ok=True)

    # 경로별 파일 개수 세기
    path_counts = defaultdict(list)
    for csv_file in raw_dir.glob('*.csv'):
        # 파일명에서 경로 추출 (예: 1_2_3.csv -> 1->2)
        parts = csv_file.stem.split('_')[:2]
        if len(parts) >= 2:
            path = f"{parts[0]}->{parts[1]}"
            path_counts[path].append(csv_file)

    print(f"📊 경로별 샘플 수 분석:")
    print(f"   총 {len(path_counts)}개 경로\n")

    # 통계
    min_samples = 3  # 최소 3개 이상 필요
    good_paths = []
    low_paths = []

    for path, files in sorted(path_counts.items()):
        count = len(files)
        if count >= min_samples:
            good_paths.append((path, files))
        else:
            low_paths.append((path, files))

    print(f"✅ 충분한 샘플 ({min_samples}개 이상): {len(good_paths)}개 경로")
    print(f"❌ 부족한 샘플 ({min_samples}개 미만): {len(low_paths)}개 경로\n")

    # 파일 이동
    moved_count = 0
    kept_count = 0

    for path, files in good_paths:
        for f in files:
            dest = filtered_dir / f.name
            shutil.copy2(str(f), str(dest))
            kept_count += 1

    for path, files in low_paths:
        for f in files:
            dest = low_sample_dir / f.name
            shutil.move(str(f), str(dest))
            moved_count += 1
        print(f"  제거: {path} ({len(files)}개)")

    print(f"\n=== 결과 ===")
    print(f"유지: {kept_count}개 파일 → data/raw_filtered/")
    print(f"제거: {moved_count}개 파일 → data/low_sample/")
    print(f"\n다음 단계:")
    print(f"1. data/raw_filtered/ → data/raw/로 이동")
    print(f"2. bash scripts/preprocess.sh 재실행")

if __name__ == '__main__':
    main()
