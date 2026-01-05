#!/usr/bin/env python3
"""
체크포인트 테스트 전용 - LSTM (Hyena 제거)

- 한 번 실행하면 EUC + MAN 둘 다 출력
- Noise robustness test도 EUC + MAN 둘 다 출력
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMPositioning
from train import (
    SlidingWindowDataset,
    denormalize_coord,
    euclidean_distance_m,
    manhattan_distance_m,
)


def summarize(dist: np.ndarray) -> dict:
    dist = dist.astype(np.float32)
    return {
        "rmse": float(np.sqrt(np.mean(dist ** 2))),
        "mae": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "min": float(np.min(dist)),
        "max": float(np.max(dist)),
        "cdf1": float(np.mean(dist <= 1.0) * 100),
        "cdf2": float(np.mean(dist <= 2.0) * 100),
        "cdf3": float(np.mean(dist <= 3.0) * 100),
        "cdf5": float(np.mean(dist <= 5.0) * 100),
    }


def test_model(
    checkpoint_path: Path,
    data_dir: Path,
    batch_size: int = 300,
    device: str = "cuda",
    noise_test: bool = True,
):
    print("=" * 80)
    print("🧪 Test Only Mode (LSTM) - EUC + MAN")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data dir:   {data_dir}")
    print()

    # meta
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    with meta_path.open() as f:
        meta = json.load(f)

    n_features = int(meta["n_features"])
    window_size = int(meta["window_size"])

    print("📊 데이터 정보:")
    print(f"   Features:    {n_features}")
    print(f"   Window size: {window_size}")
    print()

    # dataset
    test_path = data_dir / "test.jsonl"
    test_ds = SlidingWindowDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"   Test samples: {len(test_ds)}\n")

    # device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 사용")

    use_amp = device.type == "cuda"

    # load checkpoint
    print("🔄 Checkpoint 로드 중...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "epoch" in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    val_metrics = checkpoint.get("val_metrics", None)
    if isinstance(val_metrics, dict):
        e = val_metrics.get("euc", None)
        if e and "p90" in e:
            print(f"   Saved Val EUC P90: {e['p90']:.3f}m")

    cfg = checkpoint.get("config", {})
    hidden_dim = int(cfg.get("hidden_dim", 600))
    num_layers = int(cfg.get("num_layers", 3))
    dropout = float(cfg.get("dropout", 0.0))
    use_fc_relu = bool(cfg.get("use_fc_relu", False))

    model = LSTMPositioning(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_fc_relu=use_fc_relu,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("✅ 모델 로드 완료")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # test
    print("📈 Test 평가 중... (EUC + MAN)")
    dist_euc = []
    dist_man = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", ncols=110)
        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(features)
            else:
                pred = model(features)

            pred_np = pred.detach().cpu().numpy()
            tgt_np = targets.detach().cpu().numpy()

            for i in range(len(pred_np)):
                pxy = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                txy = denormalize_coord(tgt_np[i, 0], tgt_np[i, 1])
                dist_euc.append(euclidean_distance_m(pxy, txy))
                dist_man.append(manhattan_distance_m(pxy, txy))

    dist_euc = np.array(dist_euc, dtype=np.float32)
    dist_man = np.array(dist_man, dtype=np.float32)

    e = summarize(dist_euc)
    m = summarize(dist_man)

    print(
        f"\n[Test Results]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 EUC:\n"
        f"  MAE: {e['mae']:.3f}m  RMSE: {e['rmse']:.3f}m  P50: {e['median']:.3f}m  P90: {e['p90']:.3f}m  P95: {e['p95']:.3f}m\n"
        f"  Min: {e['min']:.3f}m  Max:  {e['max']:.3f}m\n"
        f"  CDF: ≤1m {e['cdf1']:.1f}% | ≤2m {e['cdf2']:.1f}% | ≤3m {e['cdf3']:.1f}% | ≤5m {e['cdf5']:.1f}%\n"
        f"\n"
        f"📌 MAN:\n"
        f"  MAE: {m['mae']:.3f}m  RMSE: {m['rmse']:.3f}m  P50: {m['median']:.3f}m  P90: {m['p90']:.3f}m  P95: {m['p95']:.3f}m\n"
        f"  Min: {m['min']:.3f}m  Max:  {m['max']:.3f}m\n"
        f"  CDF: ≤1m {m['cdf1']:.1f}% | ≤2m {m['cdf2']:.1f}% | ≤3m {m['cdf3']:.1f}% | ≤5m {m['cdf5']:.1f}%\n"
    )

    # Noise robustness
    if noise_test:
        print("\n🔊 Noise Robustness Test (EUC + MAN)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 전체 test feature std
        all_features = []
        for x, _ in test_loader:
            all_features.append(x)
        all_features = torch.cat(all_features, dim=0)
        signal_std = all_features.std().item()

        noise_percentages = [1, 5, 10, 20]
        for pct in noise_percentages:
            noise_std = signal_std * (pct / 100.0)

            noise_dist_euc = []
            noise_dist_man = []

            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(device)
                    targets = targets.to(device)

                    noise = torch.randn_like(features) * noise_std
                    noisy = features + noise

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            pred = model(noisy)
                    else:
                        pred = model(noisy)

                    pred_np = pred.detach().cpu().numpy()
                    tgt_np = targets.detach().cpu().numpy()

                    for i in range(len(pred_np)):
                        pxy = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                        txy = denormalize_coord(tgt_np[i, 0], tgt_np[i, 1])
                        noise_dist_euc.append(euclidean_distance_m(pxy, txy))
                        noise_dist_man.append(manhattan_distance_m(pxy, txy))

            noise_mae_euc = float(np.mean(noise_dist_euc))
            noise_mae_man = float(np.mean(noise_dist_man))

            deg_euc = ((noise_mae_euc - e["mae"]) / e["mae"]) * 100 if e["mae"] > 1e-9 else 0.0
            deg_man = ((noise_mae_man - m["mae"]) / m["mae"]) * 100 if m["mae"] > 1e-9 else 0.0

            print(
                f"  Noise {pct:>2d}%: "
                f"EUC MAE={noise_mae_euc:.3f}m ({deg_euc:+.1f}%) | "
                f"MAN MAE={noise_mae_man:.3f}m ({deg_man:+.1f}%)"
            )

    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a trained LSTM model from checkpoint (EUC+MAN)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-noise-test", action="store_true")

    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    test_model(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        device=device,
        noise_test=not args.no_noise_test,
    )
