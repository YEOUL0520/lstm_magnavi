#!/usr/bin/env python3
"""
나쁜 데이터를 자동으로 필터링합니다.
"""
import shutil
from pathlib import Path
import sys

# 분석 스크립트 import
sys.path.append(str(Path(__file__).parent))
from analyze_data import analyze_csv

def main():
    data_dir = Path('data/raw')
    bad_dir = Path('data/bad')  # 나쁜 데이터 이동할 디렉토리
    bad_dir.mkdir(exist_ok=True)

    csv_files = sorted(data_dir.glob('*.csv'))
    print(f"📊 총 {len(csv_files)}개 파일 분석 중...\n")

    moved_files = []

    # Thresholds
    MAX_OUTLIER = 50.0  # 자기장 이상 threshold (50 이상은 확실히 이상)
    MIN_MOVEMENT = 5.0  # 움직임 최소 threshold

    for csv_file in csv_files:
        result = analyze_csv(csv_file)
        if not result:
            continue

        should_move = False
        reasons = []

        # 1. 자기장 이상 (매우 심각)
        if result['outlier_score'] > MAX_OUTLIER:
            should_move = True
            reasons.append(f"자기장 이상 ({result['outlier_score']:.1f})")

        # 2. 움직임 거의 없음
        if result['movement'] < MIN_MOVEMENT:
            should_move = True
            reasons.append(f"움직임 없음 ({result['movement']:.2f})")

        # 3. 버튼 이벤트 없음
        if result['button_count'] == 0:
            should_move = True
            reasons.append("버튼 없음")

        # 4. 너무 짧음 (< 200)
        if result['length'] < 200:
            should_move = True
            reasons.append(f"너무 짧음 ({result['length']})")

        # 참고: 긴 시퀀스는 Hyena에 유리하므로 제거 안 함

        if should_move:
            # 파일 이동
            dest = bad_dir / csv_file.name
            shutil.move(str(csv_file), str(dest))
            moved_files.append((csv_file.name, reasons))

    # 결과 출력
    print("=" * 80)
    print(f"🗑️  나쁜 데이터 필터링 완료!")
    print("=" * 80)
    print(f"\n이동된 파일: {len(moved_files)}개")
    print(f"남은 파일: {len(csv_files) - len(moved_files)}개")
    print(f"\n나쁜 데이터 위치: {bad_dir}/")

    if moved_files:
        print(f"\n이동된 파일 샘플 (처음 10개):")
        for name, reasons in moved_files[:10]:
            print(f"  {name}: {', '.join(reasons)}")

        if len(moved_files) > 10:
            print(f"  ... 외 {len(moved_files) - 10}개")

    print("\n✅ 이제 전처리를 다시 실행하세요:")
    print("   bash scripts/preprocess.sh")

if __name__ == '__main__':
    main()
