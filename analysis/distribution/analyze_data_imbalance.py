#!/usr/bin/env python3
"""데이터 불균형 분석: 공간적 분포, 경로별 샘플 수, 이상치 위치 패턴"""
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
try:
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("⚠️ 한글 폰트 로드 실패")

# 역정규화
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0

def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (x, y)

def load_raw_csv_info(raw_dir: Path):
    """원본 CSV 파일 정보 로드"""
    csv_files = list(raw_dir.glob("*.csv"))

    path_to_files = defaultdict(list)
    for csv_file in csv_files:
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            path_id = f"{parts[0]}_{parts[1]}"
            path_to_files[path_id].append(csv_file)

    return path_to_files

def analyze_data_imbalance(data_dir: Path, raw_dir: Path = None):
    """데이터 불균형 분석"""

    print("=" * 80)
    print("🔍 데이터 불균형 분석")
    print("=" * 80)
    print()

    # 1. 전처리된 데이터 로드
    splits = ['train', 'val', 'test']
    all_data = {}

    for split in splits:
        jsonl_path = data_dir / f"{split}.jsonl"
        samples = []
        with jsonl_path.open() as f:
            for line in f:
                samples.append(json.loads(line))
        all_data[split] = samples

    # 2. 좌표 분포 분석
    print("=" * 80)
    print("📍 좌표 공간 분포 분석")
    print("=" * 80)

    for split in splits:
        samples = all_data[split]
        targets = [s['target'] for s in samples]

        # 역정규화
        coords = [denormalize_coord(t[0], t[1]) for t in targets]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        print(f"\n{split.upper()}:")
        print(f"  샘플 수: {len(samples)}")
        print(f"  X 범위: [{min(x_coords):.2f}, {max(x_coords):.2f}]m (mean={np.mean(x_coords):.2f})")
        print(f"  Y 범위: [{min(y_coords):.2f}, {max(y_coords):.2f}]m (mean={np.mean(y_coords):.2f})")

        # 건물 중심 기준 분포 (COORD_CENTER = (-41, 0) 기준)
        center_x, center_y = COORD_CENTER
        distances_from_center = [
            np.sqrt((x - center_x)**2 + (y - center_y)**2)
            for x, y in coords
        ]

        print(f"  중심으로부터 거리:")
        print(f"    평균: {np.mean(distances_from_center):.2f}m")
        print(f"    중앙값: {np.median(distances_from_center):.2f}m")
        print(f"    표준편차: {np.std(distances_from_center):.2f}m")

        # 거리 구간별 샘플 수
        bins = [0, 5, 10, 15, 20, 30, 100]
        labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-30m', '>30m']

        print(f"  거리 구간별 분포:")
        for i in range(len(bins)-1):
            count = sum(bins[i] <= d < bins[i+1] for d in distances_from_center)
            pct = count / len(distances_from_center) * 100
            print(f"    {labels[i]:<10}: {count:>5}개 ({pct:>5.1f}%)")

    print()

    # 3. 원본 CSV 기반 경로별 샘플 수 분석
    if raw_dir and raw_dir.exists():
        print("=" * 80)
        print("📊 원본 데이터 경로별 파일 수")
        print("=" * 80)

        path_to_files = load_raw_csv_info(raw_dir)

        # 경로별 파일 수 통계
        file_counts = [len(files) for files in path_to_files.values()]

        print(f"\n총 경로 수: {len(path_to_files)}")
        print(f"경로당 파일 수:")
        print(f"  평균: {np.mean(file_counts):.2f}개")
        print(f"  중앙값: {np.median(file_counts):.0f}개")
        print(f"  최소: {min(file_counts)}개")
        print(f"  최대: {max(file_counts)}개")
        print(f"  표준편차: {np.std(file_counts):.2f}개")

        # 파일 수 분포
        print(f"\n파일 수 분포:")
        file_count_dist = Counter(file_counts)
        for count in sorted(file_count_dist.keys()):
            num_paths = file_count_dist[count]
            print(f"  {count}개 파일: {num_paths}개 경로")

        # 상위/하위 10개 경로
        sorted_paths = sorted(path_to_files.items(), key=lambda x: len(x[1]), reverse=True)

        print(f"\n📈 파일 수 상위 10개 경로:")
        for path_id, files in sorted_paths[:10]:
            print(f"  {path_id}: {len(files)}개")

        print(f"\n📉 파일 수 하위 10개 경로:")
        for path_id, files in sorted_paths[-10:]:
            print(f"  {path_id}: {len(files)}개")

        print()

    # 4. 전처리 후 경로별 샘플 수 추정
    print("=" * 80)
    print("🔢 전처리 후 샘플 수 분석 (window=250, stride=50)")
    print("=" * 80)

    # meta.json에서 window size, stride 확인
    meta_path = data_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)

    window_size = meta['window_size']
    stride = meta['stride']

    print(f"\n전처리 설정:")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    print()

    # CSV 파일 길이별 샘플 생성 개수 계산
    if raw_dir and raw_dir.exists():
        import csv

        csv_files = list(raw_dir.glob("*.csv"))[:50]  # 샘플링 (50개만)
        csv_lengths = []

        for csv_file in csv_files:
            with csv_file.open() as f:
                reader = csv.DictReader(f)
                length = sum(1 for _ in reader)
                csv_lengths.append(length)

        csv_lengths = np.array(csv_lengths)

        # 예상 샘플 수 계산
        expected_samples = []
        for length in csv_lengths:
            if length >= window_size:
                n_samples = (length - window_size) // stride + 1
                expected_samples.append(n_samples)

        print(f"CSV 파일 분석 (샘플 {len(csv_files)}개):")
        print(f"  CSV 평균 길이: {np.mean(csv_lengths):.0f} 스텝")
        print(f"  CSV당 평균 샘플 생성: {np.mean(expected_samples):.1f}개")
        print(f"  샘플 생성 범위: [{min(expected_samples)}, {max(expected_samples)}]개")
        print()

        # 길이 분포
        print(f"CSV 길이 분포:")
        length_bins = [0, 300, 400, 500, 600, 1000, 10000]
        length_labels = ['<300', '300-400', '400-500', '500-600', '600-1000', '>1000']

        for i in range(len(length_bins)-1):
            count = sum(length_bins[i] <= l < length_bins[i+1] for l in csv_lengths)
            pct = count / len(csv_lengths) * 100
            print(f"  {length_labels[i]:<12}: {count:>3}개 ({pct:>5.1f}%)")
        print()

    # 5. 시각화
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 12))

    # 5-1. 좌표 공간 분포 (각 split별)
    for idx, split in enumerate(splits):
        samples = all_data[split]
        targets = [s['target'] for s in samples]
        coords = [denormalize_coord(t[0], t[1]) for t in targets]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        ax = plt.subplot(2, 3, idx + 1)

        # 2D 히스토그램
        h = ax.hist2d(x_coords, y_coords, bins=30, cmap='YlOrRd')
        ax.scatter(*COORD_CENTER, c='blue', s=200, marker='X',
                  edgecolors='black', linewidths=2, label='중심', zorder=5)
        ax.set_xlabel('X (m)', fontproperties=font_prop)
        ax.set_ylabel('Y (m)', fontproperties=font_prop)
        ax.set_title(f'{split.upper()} 좌표 분포 ({len(samples)}개)', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(h[3], ax=ax, label='샘플 수')

    # 5-2. 중심으로부터 거리 분포
    ax = plt.subplot(2, 3, 4)

    for split in splits:
        samples = all_data[split]
        targets = [s['target'] for s in samples]
        coords = [denormalize_coord(t[0], t[1]) for t in targets]

        center_x, center_y = COORD_CENTER
        distances = [
            np.sqrt((x - center_x)**2 + (y - center_y)**2)
            for x, y in coords
        ]

        ax.hist(distances, bins=30, alpha=0.5, label=split.upper(), edgecolor='black')

    ax.set_xlabel('중심으로부터 거리 (m)', fontproperties=font_prop)
    ax.set_ylabel('샘플 수', fontproperties=font_prop)
    ax.set_title('중심으로부터 거리 분포', fontproperties=font_prop)
    ax.legend(prop=font_prop)
    ax.grid(True, alpha=0.3)

    # 5-3. 경로별 파일 수 분포 (있으면)
    if raw_dir and raw_dir.exists():
        ax = plt.subplot(2, 3, 5)

        path_to_files = load_raw_csv_info(raw_dir)
        file_counts = [len(files) for files in path_to_files.values()]

        ax.hist(file_counts, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(file_counts), color='red', linestyle='--',
                  label=f'평균: {np.mean(file_counts):.1f}')
        ax.axvline(np.median(file_counts), color='blue', linestyle='--',
                  label=f'중앙값: {np.median(file_counts):.0f}')
        ax.set_xlabel('경로당 CSV 파일 수', fontproperties=font_prop)
        ax.set_ylabel('경로 수', fontproperties=font_prop)
        ax.set_title('경로별 원본 파일 수 분포', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        ax.grid(True, alpha=0.3)

    # 5-4. Split별 샘플 수
    ax = plt.subplot(2, 3, 6)

    split_counts = [len(all_data[s]) for s in splits]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(splits, split_counts, color=colors, edgecolor='black', alpha=0.7)

    # 막대 위에 숫자 표시
    for bar, count in zip(bars, split_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}',
               ha='center', va='bottom', fontproperties=font_prop, fontsize=12)

    ax.set_ylabel('샘플 수', fontproperties=font_prop)
    ax.set_title('Split별 샘플 수', fontproperties=font_prop)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "data_imbalance_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 시각화 저장: {output_path}")
    print()

    # 6. 결론 및 권장사항
    print("=" * 80)
    print("📋 분석 결론 및 권장사항")
    print("=" * 80)

    # 데이터 불균형 여부 판단
    all_samples = []
    for split in splits:
        samples = all_data[split]
        targets = [s['target'] for s in samples]
        coords = [denormalize_coord(t[0], t[1]) for t in targets]
        all_samples.extend(coords)

    center_x, center_y = COORD_CENTER
    all_distances = [
        np.sqrt((x - center_x)**2 + (y - center_y)**2)
        for x, y in all_samples
    ]

    # 중심 집중도 측정 (중심 10m 이내 비율)
    central_ratio = sum(d <= 10 for d in all_distances) / len(all_distances)

    print()
    if central_ratio > 0.3:
        print(f"⚠️ 중심 집중도 높음: {central_ratio*100:.1f}%가 중심 10m 이내")
        print("  → 건물 중심 데이터 과다")
        print("  → 외곽 영역 예측 성능 저하 가능")
        print()
        print("💡 권장 사항:")
        print("  1. 경로별 샘플링 (각 경로에서 동일한 샘플 수 추출)")
        print("  2. 공간별 가중치 (외곽 영역 샘플에 높은 가중치)")
        print("  3. 데이터 증강 (외곽 경로 데이터 augmentation)")
    else:
        print(f"✅ 데이터 분포 양호: 중심 집중도 {central_ratio*100:.1f}%")

    print()

    # 경로별 불균형
    if raw_dir and raw_dir.exists():
        path_to_files = load_raw_csv_info(raw_dir)
        file_counts = [len(files) for files in path_to_files.values()]

        max_count = max(file_counts)
        min_count = min(file_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 3:
            print(f"⚠️ 경로별 파일 수 불균형: {imbalance_ratio:.1f}x 차이")
            print(f"  최대: {max_count}개, 최소: {min_count}개")
            print()
            print("💡 권장 사항:")
            print("  1. 경로별 균등 샘플링 (--split-mode를 'balanced'로)")
            print("  2. 적은 경로는 augmentation으로 보강")
        else:
            print(f"✅ 경로별 균형 양호: 최대/최소 비율 {imbalance_ratio:.1f}x")

    print()
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze data imbalance")
    parser.add_argument("--data-dir", type=str, default="data/sliding_mag4")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                       help="원본 CSV 디렉토리 (경로별 분석용)")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    analyze_data_imbalance(
        data_dir=Path(args.data_dir),
        raw_dir=raw_dir
    )
