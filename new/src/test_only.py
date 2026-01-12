#!/usr/bin/env python3
"""체크포인트를 사용한 테스트 전용 스크립트"""
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys

# 역정규화
COORD_CENTER = (-44.3, -0.3)
COORD_SCALE = 48.8

def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (x, y)

# Dataset import
sys.path.append(str(Path(__file__).parent))
from train_sliding import SlidingWindowDataset
from model import HyenaPositioning

def test_model(
    checkpoint_path: Path,
    data_dir: Path,
    batch_size: int = 32,
    hidden_dim: int = 256,
    depth: int = 8,
    device: str = "cuda",
    noise_test: bool = True,
):
    """체크포인트로 모델 테스트"""

    print("=" * 80)
    print("🧪 Test Only Mode")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data dir: {data_dir}")
    print()

    # 메타데이터 로드
    meta_path = data_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)

    n_features = meta["n_features"]
    window_size = meta["window_size"]

    print(f"📊 데이터 정보:")
    print(f"   Features: {n_features}")
    print(f"   Window size: {window_size}")
    print()

    # 테스트 데이터
    test_path = data_dir / "test.jsonl"
    test_ds = SlidingWindowDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"   Test: {len(test_ds)}개 샘플")
    print()

    # Device 설정
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 사용")

    # 모델 로드
    print(f"🔄 Checkpoint 로드 중...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 체크포인트 정보 출력
        if "epoch" in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if "val_rmse" in checkpoint:
            print(f"   Val RMSE: {checkpoint['val_rmse']:.3f}m")
        if "val_p90" in checkpoint:
            print(f"   Val P90: {checkpoint['val_p90']:.3f}m")

        # 메타 정보 (checkpoint 우선, 없으면 data meta 사용)
        if "meta" in checkpoint:
            checkpoint_meta = checkpoint["meta"]
            n_features = checkpoint_meta.get("n_features", n_features)
            window_size = checkpoint_meta.get("window_size", window_size)

    except Exception as e:
        print(f"❌ Checkpoint 로드 실패: {e}")
        return

    # 모델 생성
    model = HyenaPositioning(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=2,
        depth=depth,
        dropout=0.1,
        num_edge_types=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ 모델 로드 완료")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # AMP 설정
    use_amp = device.type == 'cuda'

    # Test 평가
    print("📈 Test 평가 중...")
    test_distances = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", ncols=100)
        for features, targets in test_pbar:
            features = features.to(device)
            targets = targets.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
                    outputs = model(features, edge_ids)
                    pred = outputs[:, -1, :]
            else:
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
                test_distances.append(dist)

    test_distances_array = np.array(test_distances)

    # 기본 메트릭
    test_rmse = np.sqrt(np.mean(test_distances_array ** 2))
    test_mae = np.mean(test_distances_array)

    # Percentiles
    test_median = np.median(test_distances_array)
    test_p90 = np.percentile(test_distances_array, 90)
    test_p95 = np.percentile(test_distances_array, 95)
    test_max = np.max(test_distances_array)
    test_min = np.min(test_distances_array)

    # CDF
    cdf_1m = np.mean(test_distances_array <= 1.0) * 100
    cdf_2m = np.mean(test_distances_array <= 2.0) * 100
    cdf_3m = np.mean(test_distances_array <= 3.0) * 100
    cdf_5m = np.mean(test_distances_array <= 5.0) * 100

    print(
        f"\n[Test Results]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 기본 메트릭:\n"
        f"  MAE (Mean Absolute Error):  {test_mae:.3f}m\n"
        f"  RMSE (Root Mean Square):    {test_rmse:.3f}m\n"
        f"\n"
        f"📈 분포:\n"
        f"  Median (P50):  {test_median:.3f}m\n"
        f"  P90:           {test_p90:.3f}m\n"
        f"  P95:           {test_p95:.3f}m\n"
        f"  Min:           {test_min:.3f}m\n"
        f"  Max:           {test_max:.3f}m\n"
        f"\n"
        f"📍 CDF (누적 분포):\n"
        f"  ≤ 1m:  {cdf_1m:.1f}%\n"
        f"  ≤ 2m:  {cdf_2m:.1f}%\n"
        f"  ≤ 3m:  {cdf_3m:.1f}%\n"
        f"  ≤ 5m:  {cdf_5m:.1f}%\n"
    )

    # Noise Robustness Test
    if noise_test:
        print(f"\n🔊 Noise Robustness Test:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 신호 표준편차 계산 (전체 test 데이터)
        all_features = []
        for features, _ in test_loader:
            all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        signal_std = all_features.std().item()

        # 퍼센트 기반 노이즈 레벨 (신호 대비 상대적)
        noise_percentages = [1, 5, 10, 20]

        for noise_pct in noise_percentages:
            noise_std = signal_std * (noise_pct / 100.0)
            noise_distances = []

            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(device)
                    targets = targets.to(device)

                    # Add Gaussian noise (percentage of signal)
                    noise = torch.randn_like(features) * noise_std
                    noisy_features = features + noise

                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            edge_ids = torch.zeros(noisy_features.size(0), dtype=torch.long, device=device)
                            outputs = model(noisy_features, edge_ids)
                            pred = outputs[:, -1, :]
                    else:
                        edge_ids = torch.zeros(noisy_features.size(0), dtype=torch.long, device=device)
                        outputs = model(noisy_features, edge_ids)
                        pred = outputs[:, -1, :]

                    pred_np = pred.cpu().numpy()
                    target_np = targets.cpu().numpy()

                    for i in range(len(pred_np)):
                        pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                        target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                        dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])
                        noise_distances.append(dist)

            noise_mae = np.mean(noise_distances)
            degradation = ((noise_mae - test_mae) / test_mae) * 100

            print(f"  Noise {noise_pct:>2d}%: MAE={noise_mae:.3f}m (degradation: {degradation:+.1f}%)")

    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a trained model from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, required=True, help="Test data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=8, help="Number of Hyena blocks")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--no-noise-test", action="store_true", help="Skip noise robustness test")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    test_model(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=device,
        noise_test=not args.no_noise_test,
    )
