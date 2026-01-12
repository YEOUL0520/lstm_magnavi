#!/usr/bin/env python3
"""큰 오차를 보이는 패턴 분석"""
import torch
import json
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_sliding import SlidingWindowDataset
from model import HyenaPositioning

def analyze_outliers(checkpoint_path: Path, data_dir: Path, threshold: float = 5.0):
    """threshold 이상의 오차를 보이는 샘플 분석"""

    # 메타데이터 로드
    with (data_dir / "meta.json").open() as f:
        meta = json.load(f)

    # 모델 로드
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HyenaPositioning(
        input_dim=meta["n_features"],
        hidden_dim=384,
        depth=10
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 테스트 데이터 로드
    test_dataset = SlidingWindowDataset(data_dir / "test.jsonl")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # Outlier 수집
    outliers = []
    all_errors = []
    all_positions = []  # 모든 샘플의 실제 위치

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_loader):
            features = features.to(device)
            targets = targets.to(device)

            # edge_ids 생성 (모두 0으로)
            edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)

            outputs = model(features, edge_ids)  # (batch, seq_len, 2)
            preds = outputs[:, -1, :]  # 마지막 타임스텝만 사용 (batch, 2)

            # 역정규화
            COORD_CENTER = torch.tensor([-41.0, 0.0], device=device)
            COORD_SCALE = 50.0

            preds_real = preds * COORD_SCALE + COORD_CENTER
            targets_real = targets * COORD_SCALE + COORD_CENTER

            # 유클리드 거리
            distances = torch.norm(preds_real - targets_real, dim=1)

            for i, dist in enumerate(distances):
                sample_idx = batch_idx * 64 + i
                error = dist.item()
                all_errors.append(error)

                # 모든 샘플의 위치 저장
                all_positions.append({
                    "x": targets_real[i, 0].item(),
                    "y": targets_real[i, 1].item(),
                    "error": error
                })

                if error > threshold:
                    outliers.append({
                        "sample_idx": sample_idx,
                        "error": error,
                        "pred": preds_real[i].cpu().tolist(),
                        "target": targets_real[i].cpu().tolist(),
                        "pred_norm": preds[i].cpu().tolist(),
                        "target_norm": targets[i].cpu().tolist()
                    })

    # 통계
    print("=" * 80)
    print("🔍 Outlier 분석")
    print("=" * 80)
    print(f"총 샘플: {len(all_errors)}")
    print(f"Outliers (>{threshold}m): {len(outliers)} ({len(outliers)/len(all_errors)*100:.1f}%)")
    print(f"최대 오차: {max(all_errors):.3f}m")
    print()

    # 상위 10개 outlier
    outliers_sorted = sorted(outliers, key=lambda x: x["error"], reverse=True)
    print("📊 상위 10개 Outlier:")
    print("-" * 80)
    for i, out in enumerate(outliers_sorted[:10], 1):
        print(f"{i}. Sample {out['sample_idx']}: Error={out['error']:.3f}m")
        print(f"   Target: ({out['target'][0]:.2f}, {out['target'][1]:.2f})")
        print(f"   Pred:   ({out['pred'][0]:.2f}, {out['pred'][1]:.2f})")
        print(f"   Norm Target: ({out['target_norm'][0]:.4f}, {out['target_norm'][1]:.4f})")
        print(f"   Norm Pred:   ({out['pred_norm'][0]:.4f}, {out['pred_norm'][1]:.4f})")
        print()

    # 구간별 전체 데이터 분포
    print("📊 구간별 데이터 분포 vs Outlier 비율:")
    print("-" * 80)

    # X 좌표 구간별 통계
    x_bin_total = defaultdict(int)
    x_bin_outliers = defaultdict(int)

    for pos in all_positions:
        x = pos["x"]
        x_bin = int(x // 10) * 10
        x_bin_total[x_bin] += 1
        if pos["error"] > threshold:
            x_bin_outliers[x_bin] += 1

    print("X 좌표 분포 (10m 단위):")
    print(f"{'구간':<20} {'전체':<8} {'Outlier':<10} {'비율':<10} {'그래프'}")
    print("-" * 80)
    for x_bin in sorted(x_bin_total.keys()):
        total = x_bin_total[x_bin]
        outlier_count = x_bin_outliers[x_bin]
        ratio = (outlier_count / total * 100) if total > 0 else 0
        bar = "█" * int(ratio // 2)
        print(f"  {x_bin:>4}m ~ {x_bin+10:>4}m: {total:>6}개  {outlier_count:>3}개  {ratio:>6.2f}%  {bar}")

    print()

    # Y 좌표 구간별 통계
    y_bin_total = defaultdict(int)
    y_bin_outliers = defaultdict(int)

    for pos in all_positions:
        y = pos["y"]
        y_bin = int(y // 5) * 5
        y_bin_total[y_bin] += 1
        if pos["error"] > threshold:
            y_bin_outliers[y_bin] += 1

    print("Y 좌표 분포 (5m 단위):")
    print(f"{'구간':<20} {'전체':<8} {'Outlier':<10} {'비율':<10} {'그래프'}")
    print("-" * 80)
    for y_bin in sorted(y_bin_total.keys()):
        total = y_bin_total[y_bin]
        outlier_count = y_bin_outliers[y_bin]
        ratio = (outlier_count / total * 100) if total > 0 else 0
        bar = "█" * int(ratio // 2)
        print(f"  {y_bin:>4}m ~ {y_bin+5:>4}m: {total:>6}개  {outlier_count:>3}개  {ratio:>6.2f}%  {bar}")

    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/hyena_mag4/checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data/sliding_mag4")
    parser.add_argument("--threshold", type=float, default=5.0, help="Outlier 기준 (미터)")

    args = parser.parse_args()

    analyze_outliers(
        Path(args.checkpoint),
        Path(args.data_dir),
        args.threshold
    )
