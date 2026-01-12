#!/usr/bin/env python3
"""
데이터 품질 분석 스크립트
나쁜 데이터를 찾아냅니다.
"""
import csv
import sys
from pathlib import Path
import numpy as np

def analyze_csv(csv_path):
    """CSV 파일의 품질 지표 분석"""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return None

    # 기본 정보
    length = len(rows)

    # 자기장 통계
    mag_x = [float(row['MagX']) for row in rows]
    mag_y = [float(row['MagY']) for row in rows]
    mag_z = [float(row['MagZ']) for row in rows]

    # 버튼 이벤트 수
    button_count = 0
    if 'Highlighted' in rows[0] or 'RightAngle' in rows[0]:
        for row in rows:
            if row.get('Highlighted', '').strip().lower() in ('1', 'true') or \
               row.get('RightAngle', '').strip().lower() in ('1', 'true'):
                button_count += 1

    # 움직임 분석 (자기장 변화)
    mag_x_std = np.std(mag_x)
    mag_y_std = np.std(mag_y)
    mag_z_std = np.std(mag_z)
    movement = mag_x_std + mag_y_std + mag_z_std

    # 이상치 감지 (평균에서 너무 멀리 떨어진 값)
    mag_x_mean = np.mean(mag_x)
    mag_y_mean = np.mean(mag_y)
    mag_z_mean = np.mean(mag_z)

    outlier_x = abs(mag_x_mean - (-33.0))  # BASE_MAG 기준
    outlier_y = abs(mag_y_mean - (-15.0))
    outlier_z = abs(mag_z_mean - (-42.0))
    outlier_score = outlier_x + outlier_y + outlier_z

    return {
        'path': csv_path,
        'length': length,
        'button_count': button_count,
        'movement': movement,
        'outlier_score': outlier_score,
        'mag_x_mean': mag_x_mean,
        'mag_y_mean': mag_y_mean,
        'mag_z_mean': mag_z_mean,
    }

def main():
    data_dir = Path('data/raw')
    csv_files = sorted(data_dir.glob('*.csv'))

    print(f"📊 총 {len(csv_files)}개 CSV 파일 분석 중...\n")

    results = []
    for csv_file in csv_files:
        result = analyze_csv(csv_file)
        if result:
            results.append(result)

    if not results:
        print("❌ 분석할 데이터가 없습니다.")
        return

    # 문제 데이터 탐지
    print("=" * 80)
    print("⚠️  의심스러운 데이터 (다음 조건 중 하나라도 해당)")
    print("=" * 80)

    suspicious = []

    for r in results:
        issues = []

        # 1. 너무 짧음 (< 200)
        if r['length'] < 200:
            issues.append(f"너무 짧음 ({r['length']} 타임스텝)")

        # 2. 너무 김 (> 5000)
        if r['length'] > 5000:
            issues.append(f"너무 김 ({r['length']} 타임스텝)")

        # 3. 버튼 이벤트 없음
        if r['button_count'] == 0:
            issues.append("버튼 이벤트 없음")

        # 4. 움직임 거의 없음 (정지 상태)
        if r['movement'] < 5.0:  # threshold
            issues.append(f"움직임 없음 (score={r['movement']:.2f})")

        # 5. 자기장 이상치 (평균에서 20 이상 차이)
        if r['outlier_score'] > 20.0:
            issues.append(f"자기장 이상 (score={r['outlier_score']:.1f})")

        if issues:
            suspicious.append((r, issues))

    if suspicious:
        for r, issues in suspicious:
            print(f"\n📁 {r['path'].name}")
            for issue in issues:
                print(f"   ❌ {issue}")
            print(f"   Stats: len={r['length']}, btn={r['button_count']}, "
                  f"mov={r['movement']:.1f}, out={r['outlier_score']:.1f}")
    else:
        print("\n✅ 의심스러운 데이터가 없습니다!")

    print("\n" + "=" * 80)
    print(f"총 {len(suspicious)}개 의심 파일 / {len(results)}개 전체 파일")
    print("=" * 80)

    # 통계 요약
    print("\n📈 전체 데이터 통계:")
    lengths = [r['length'] for r in results]
    movements = [r['movement'] for r in results]
    print(f"   길이 평균: {np.mean(lengths):.0f} (범위: {min(lengths)} ~ {max(lengths)})")
    print(f"   움직임 평균: {np.mean(movements):.2f}")
    print(f"   버튼 없는 파일: {sum(1 for r in results if r['button_count'] == 0)}개")

if __name__ == '__main__':
    main()
