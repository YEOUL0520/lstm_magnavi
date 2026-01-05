#!/usr/bin/env python3
"""아웃라이어 분석: 큰 오차를 보이는 샘플 찾기 및 분석"""
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (OS 자동 감지)
def setup_korean_font():
    """운영체제에 맞는 한글 폰트 자동 설정"""
    import platform

    system = platform.system()

    # 시스템별 한글 폰트 후보
    font_candidates = []

    if system == 'Darwin':  # macOS
        font_candidates = [
            'AppleGothic',
            'Apple SD Gothic Neo',
            'NanumGothic',
        ]
    elif system == 'Windows':
        font_candidates = [
            'Malgun Gothic',
            'NanumGothic',
            'Gulim',
        ]
    else:  # Linux
        font_candidates = [
            'NanumGothic',
            'NanumBarunGothic',
            'UnDotum',
            'DejaVu Sans',
        ]

    # 사용 가능한 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 후보 중 사용 가능한 첫 번째 폰트 찾기
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 한글 폰트 설정: {font}")
            return True

    # 한글 폰트를 찾지 못한 경우
    print("⚠️  한글 폰트를 찾을 수 없습니다. 기본 폰트로 표시됩니다.")
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 한글 폰트 설정 시도
setup_korean_font()

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))
from model import HyenaPositioning
from torch.utils.data import Dataset, DataLoader

# 역정규화
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0

def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (x, y)

class SlidingWindowDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.samples = []
        with jsonl_path.open() as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample["features"], dtype=torch.float32)
        target = torch.tensor(sample["target"], dtype=torch.float32)
        return features, target

def analyze_outliers(
    checkpoint_path: Path,
    data_dir: Path,
    threshold: float = 3.0,  # 3m 이상을 아웃라이어로 정의
    device: str = "cpu",
):
    """아웃라이어 분석"""

    print("=" * 80)
    print("🔍 아웃라이어 분석")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data dir: {data_dir}")
    print(f"  Outlier threshold: {threshold}m")
    print()

    # 메타데이터 로드
    meta_path = data_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)

    n_features = meta["n_features"]

    # 테스트 데이터
    test_path = data_dir / "test.jsonl"
    test_ds = SlidingWindowDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"📊 Test samples: {len(test_ds)}개")
    print()

    # Device 설정
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device("cpu")
        print("💻 CPU 사용")

    # 모델 로드
    print(f"🔄 Checkpoint 로드 중...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = HyenaPositioning(
        input_dim=n_features,
        hidden_dim=384,
        output_dim=2,
        depth=10,
        dropout=0.1,
        num_edge_types=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("✅ 모델 로드 완료")
    print()

    # 예측 및 오차 계산
    print("📈 예측 중...")
    all_errors = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)

            edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
            outputs = model(features, edge_ids)
            pred = outputs[:, -1, :]

            pred_np = pred.cpu().numpy()
            target_np = targets.cpu().numpy()

            for i in range(len(pred_np)):
                pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])

                # Manhattan distance
                dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])

                all_errors.append(dist)
                all_predictions.append(pred_pos)
                all_targets.append(target_pos)

    all_errors = np.array(all_errors)
    print(f"✅ 예측 완료")
    print()

    # 아웃라이어 찾기
    outlier_indices = np.where(all_errors >= threshold)[0]
    print("=" * 80)
    print(f"🎯 아웃라이어 분석 (오차 ≥ {threshold}m)")
    print("=" * 80)
    print(f"  총 샘플 수: {len(all_errors)}개")
    print(f"  아웃라이어: {len(outlier_indices)}개 ({len(outlier_indices)/len(all_errors)*100:.1f}%)")
    print()

    if len(outlier_indices) == 0:
        print("✅ 아웃라이어 없음!")
        return

    # 아웃라이어 상세 정보
    print("📋 아웃라이어 상세 정보:")
    print("-" * 80)
    print(f"{'Index':<8} {'Error(m)':<12} {'Pred(x,y)':<25} {'Target(x,y)':<25}")
    print("-" * 80)

    # 오차 큰 순으로 정렬
    sorted_indices = outlier_indices[np.argsort(all_errors[outlier_indices])[::-1]]

    # 모든 outlier 출력
    for idx in sorted_indices:
        error = all_errors[idx]
        pred = all_predictions[idx]
        target = all_targets[idx]
        print(f"{idx:<8} {error:<12.3f} ({pred[0]:>6.2f}, {pred[1]:>6.2f})      ({target[0]:>6.2f}, {target[1]:>6.2f})")

    print()

    # 통계
    print("=" * 80)
    print("📊 아웃라이어 통계")
    print("=" * 80)
    outlier_errors = all_errors[outlier_indices]
    print(f"  평균 오차: {np.mean(outlier_errors):.3f}m")
    print(f"  중앙값:    {np.median(outlier_errors):.3f}m")
    print(f"  최소:      {np.min(outlier_errors):.3f}m")
    print(f"  최대:      {np.max(outlier_errors):.3f}m")
    print()

    # 전체 오차 분포
    print("=" * 80)
    print("📈 전체 오차 분포")
    print("=" * 80)
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.5, 100]
    for p in percentiles:
        val = np.percentile(all_errors, p)
        print(f"  P{p:>5}: {val:>8.3f}m")
    print()

    # 오차 구간별 분포
    print("=" * 80)
    print("📍 오차 구간별 분포")
    print("=" * 80)
    bins = [0, 1, 2, 3, 5, 10, 20, float('inf')]
    labels = ['0-1m', '1-2m', '2-3m', '3-5m', '5-10m', '10-20m', '>20m']

    for i in range(len(bins)-1):
        count = np.sum((all_errors >= bins[i]) & (all_errors < bins[i+1]))
        pct = count / len(all_errors) * 100
        print(f"  {labels[i]:<10}: {count:>5}개 ({pct:>5.1f}%)")
    print()

    # 시각화
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 1. 오차 분포 히스토그램
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label=f'임계값 ({threshold}m)')
    plt.xlabel('오차 (m)')
    plt.ylabel('개수')
    plt.title('전체 오차 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 오차 분포 (5m 이하만)
    plt.subplot(2, 2, 2)
    plt.hist(all_errors[all_errors <= 5], bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('오차 (m)')
    plt.ylabel('개수')
    plt.title(f'정상 범위 (≤{threshold}m)')
    plt.grid(True, alpha=0.3)

    # 3. CDF (누적 분포)
    plt.subplot(2, 2, 3)
    sorted_errors = np.sort(all_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cdf, linewidth=2)
    plt.axvline(threshold, color='red', linestyle='--', label=f'{threshold}m')
    plt.axhline(90, color='orange', linestyle='--', label='P90')
    plt.axhline(95, color='blue', linestyle='--', label='P95')
    plt.xlabel('오차 (m)')
    plt.ylabel('누적 비율 (%)')
    plt.title('누적 분포 함수 (CDF)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(10, np.max(all_errors)))

    # 4. Box plot
    plt.subplot(2, 2, 4)
    plt.boxplot([all_errors[all_errors <= 5], outlier_errors],
                tick_labels=[f'정상 (≤{threshold}m)', f'이상치 (≥{threshold}m)'])
    plt.ylabel('오차 (m)')
    plt.title('오차 분포 비교')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "outlier_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 시각화 저장: {output_path}")
    print()

    # 예측 vs 실제 위치 (아웃라이어만)
    if len(outlier_indices) > 0:
        plt.figure(figsize=(10, 10))

        # 정상 샘플 (회색)
        normal_indices = np.where(all_errors < threshold)[0]
        normal_preds = [all_predictions[i] for i in normal_indices]
        normal_targets = [all_targets[i] for i in normal_indices]

        if len(normal_preds) > 0:
            normal_pred_x = [p[0] for p in normal_preds]
            normal_pred_y = [p[1] for p in normal_preds]
            normal_target_x = [t[0] for t in normal_targets]
            normal_target_y = [t[1] for t in normal_targets]

            plt.scatter(normal_target_x, normal_target_y, c='lightgray', s=20, alpha=0.3, label='정상 (실제)')
            plt.scatter(normal_pred_x, normal_pred_y, c='lightblue', s=20, alpha=0.3, label='정상 (예측)')

        # 아웃라이어 (빨강)
        outlier_preds = [all_predictions[i] for i in outlier_indices]
        outlier_targets = [all_targets[i] for i in outlier_indices]

        outlier_pred_x = [p[0] for p in outlier_preds]
        outlier_pred_y = [p[1] for p in outlier_preds]
        outlier_target_x = [t[0] for t in outlier_targets]
        outlier_target_y = [t[1] for t in outlier_targets]

        plt.scatter(outlier_target_x, outlier_target_y, c='red', s=100, marker='o',
                   edgecolors='darkred', linewidths=2, label='이상치 (실제)', zorder=5)
        plt.scatter(outlier_pred_x, outlier_pred_y, c='orange', s=100, marker='x',
                   linewidths=3, label='이상치 (예측)', zorder=5)

        # 화살표로 연결
        for i in range(len(outlier_indices)):
            plt.arrow(outlier_target_x[i], outlier_target_y[i],
                     outlier_pred_x[i] - outlier_target_x[i],
                     outlier_pred_y[i] - outlier_target_y[i],
                     color='red', alpha=0.3, width=0.1, head_width=0.5,
                     length_includes_head=True, zorder=4)

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'예측 vs 실제 위치 (이상치: {len(outlier_indices)}개)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        output_path2 = output_dir / "outlier_positions.png"
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"📊 위치 시각화 저장: {output_path2}")
        print()

    print("=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze outliers in predictions")
    parser.add_argument("--checkpoint", type=str, default="models/hyena_mag4/checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/sliding_mag4")
    parser.add_argument("--threshold", type=float, default=3.0, help="Outlier threshold in meters")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    analyze_outliers(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        threshold=args.threshold,
        device=device,
    )
