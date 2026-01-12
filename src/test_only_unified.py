#!/usr/bin/env python3
"""
Unified test_only for LSTM + Hyena
- Same test.jsonl, same metrics: EUC + MAN + CDF + Noise robustness
- LSTM output: (B,2)
- Hyena output: (B,T,2) -> use last timestep (B,2)
- IMPORTANT: Hyena is evaluated with edge_ids=zeros (same as training code)

[NEW]
- Save results to JSON + append to CSV:
  --save-dir exp_results/scarcity/_eval
  --tag 100pct
"""

import json
import csv
from pathlib import Path
import argparse
import inspect
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ✅ src 폴더를 import 경로에 추가 (python src/... 실행 안정화)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ====== 좌표 역정규화 (네가 쓰던 값 그대로) ======
COORD_CENTER = (-44.3, -0.3)
COORD_SCALE = 48.8


def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (float(x), float(y))


def euclidean_distance_m(pred_xy, true_xy) -> float:
    dx = pred_xy[0] - true_xy[0]
    dy = pred_xy[1] - true_xy[1]
    return float(np.sqrt(dx * dx + dy * dy))


def manhattan_distance_m(pred_xy, true_xy) -> float:
    return abs(pred_xy[0] - true_xy[0]) + abs(pred_xy[1] - true_xy[1])


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


class JsonlDataset(Dataset):
    """each line: {"features":[T,F], "target":[2]}"""

    def __init__(self, jsonl_path: Path):
        self.samples = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        if not self.samples:
            raise RuntimeError(f"Empty dataset: {jsonl_path}")

        self.n_features = len(self.samples[0]["features"][0])
        self.window_size = len(self.samples[0]["features"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["features"], dtype=torch.float32)  # [T,F]
        y = torch.tensor(s["target"], dtype=torch.float32)    # [2]
        return x, y


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # state dict key 대응
    state_key = None
    for k in ["model_state", "state_dict", "model", "model_state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            state_key = k
            break

    if state_key is None:
        # 가끔 ckpt 자체가 state_dict인 경우도 있음
        if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, {}, ckpt
        raise KeyError(f"Cannot find model state in checkpoint keys: {list(ckpt.keys())}")

    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    return ckpt, cfg, ckpt[state_key]


def build_model(arch: str, n_features: int, cfg: dict):
    arch = arch.lower()
    if arch == "lstm":
        # ✅ LSTMPositioning은 src/model.py
        from model import LSTMPositioning

        hidden_dim = int(cfg.get("hidden_dim", 600))
        num_layers = int(cfg.get("num_layers", 3))
        dropout = float(cfg.get("dropout", 0.0))
        use_fc_relu = bool(cfg.get("use_fc_relu", False))

        return LSTMPositioning(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_fc_relu=use_fc_relu,
        )

    elif arch == "hyena":
        # ✅ HyenaPositioning은 src/model_h.py
        from model_h import HyenaPositioning

        hidden_dim = int(cfg.get("hidden_dim", 256))
        depth = int(cfg.get("depth", 8))
        order = int(cfg.get("order", 2))
        dropout = float(cfg.get("dropout", 0.1))
        num_edge_types = int(cfg.get("num_edge_types", 1))

        return HyenaPositioning(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=2,
            depth=depth,
            order=order,
            dropout=dropout,
            num_edge_types=num_edge_types,
        )

    else:
        raise ValueError("--arch must be one of: lstm, hyena")


def hyena_accepts_edge_ids(model) -> bool:
    """Hyena forward가 (x, edge_ids=None) 형태인지 확인"""
    try:
        sig = inspect.signature(model.forward)
        return "edge_ids" in sig.parameters
    except Exception:
        return True  # 보수적으로 True


def forward_pred_xy(arch: str, model, features: torch.Tensor, use_amp: bool):
    """
    returns pred: (B,2)
    """
    arch = arch.lower()

    if arch == "lstm":
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(features)  # (B,2)
        else:
            out = model(features)
        return out

    elif arch == "hyena":
        # 학습 코드와 동일하게 edge_ids=zeros 전달
        edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=features.device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                if hyena_accepts_edge_ids(model):
                    out = model(features, edge_ids)  # (B,T,2)
                else:
                    out = model(features)            # fallback
        else:
            if hyena_accepts_edge_ids(model):
                out = model(features, edge_ids)
            else:
                out = model(features)

        # (B,T,2) -> last timestep
        return out[:, -1, :]

    else:
        raise ValueError("unknown arch")


def save_results(save_dir: Path, arch: str, tag: str, checkpoint: Path, data_dir: Path,
                 n_test: int, window: int, n_features: int,
                 e: dict, m: dict, noise_mode: str, noise_levels: str):
    """
    Save per-run JSON + append CSV.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "arch": arch,
        "tag": tag,
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "n_test": int(n_test),
        "window": int(window),
        "n_features": int(n_features),

        "euc_mae": e["mae"],
        "euc_rmse": e["rmse"],
        "euc_p50": e["median"],
        "euc_p90": e["p90"],
        "euc_p95": e["p95"],
        "euc_min": e["min"],
        "euc_max": e["max"],
        "euc_cdf_1m": e["cdf1"],
        "euc_cdf_2m": e["cdf2"],
        "euc_cdf_3m": e["cdf3"],
        "euc_cdf_5m": e["cdf5"],

        "man_mae": m["mae"],
        "man_rmse": m["rmse"],
        "man_p50": m["median"],
        "man_p90": m["p90"],
        "man_p95": m["p95"],
        "man_min": m["min"],
        "man_max": m["max"],
        "man_cdf_1m": m["cdf1"],
        "man_cdf_2m": m["cdf2"],
        "man_cdf_3m": m["cdf3"],
        "man_cdf_5m": m["cdf5"],

        "noise_mode": noise_mode,
        "noise_levels": noise_levels,
    }

    # JSON 저장
    safe_tag = tag if tag else "run"
    json_path = save_dir / f"{arch}_{safe_tag}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # CSV 누적 저장
    csv_path = save_dir / "results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print(f"💾 Saved JSON: {json_path}")
    print(f"📊 Updated CSV: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["lstm", "hyena"])
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data-dir", required=True, type=str)
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-noise-test", action="store_true")

    # ✅ 저장 옵션 추가
    ap.add_argument("--save-dir", type=str, default=None,
                    help="Directory to save results (json + csv). Example: exp_results/scarcity/_eval")
    ap.add_argument("--tag", type=str, default="",
                    help="Tag for this run (e.g., 100pct, 050pct). Used for filename and CSV.")

    # noise 옵션: percent 기반 + sigma 기반
    ap.add_argument("--noise-mode", choices=["percent", "sigma"], default="percent")
    ap.add_argument("--noise-levels", type=str, default="1,5,10,20",
                    help="percent: 1,5,10,20 / sigma: 0.1,0.2,0.5")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)

    # device
    if (not args.cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and (not args.cpu):
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU 사용")

    use_amp = (device.type == "cuda")

    # dataset
    test_path = data_dir / "test.jsonl"
    ds = JsonlDataset(test_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    n_features = ds.n_features
    window_size = ds.window_size

    print("=" * 80)
    print(f"🧪 Test Only (Unified) | arch={args.arch.upper()} | EUC + MAN + CDF + Noise")
    print("=" * 80)
    print(f"  checkpoint: {ckpt_path}")
    print(f"  data-dir:   {data_dir}")
    print(f"  test:       {len(ds)} samples | window={window_size} | features={n_features}")
    if args.tag:
        print(f"  tag:        {args.tag}")
    if args.save_dir:
        print(f"  save-dir:   {args.save_dir}")
    print()

    # load model
    ckpt, cfg, state = load_checkpoint(ckpt_path, device)
    model = build_model(args.arch, n_features, cfg).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    print("✅ 모델 로드 완료")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    if missing:
        print(f"   ⚠️ missing keys: {len(missing)} (예: {missing[:3]})")
    if unexpected:
        print(f"   ⚠️ unexpected keys: {len(unexpected)} (예: {unexpected[:3]})")
    print()

    # test
    print("📈 Test 평가 중... (EUC + MAN)")
    dist_euc, dist_man = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing", ncols=110)
        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device)

            pred_xy = forward_pred_xy(args.arch, model, features, use_amp=use_amp)

            pred_np = pred_xy.detach().cpu().numpy()
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
        "\n[Test Results]\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 EUC:\n"
        f"  MAE={e['mae']:.3f}m  RMSE={e['rmse']:.3f}m\n"
        f"  P50={e['median']:.3f}m  P90={e['p90']:.3f}m  P95={e['p95']:.3f}m\n"
        f"  Min={e['min']:.3f}m  Max={e['max']:.3f}m\n"
        f"  CDF: ≤ 1m {e['cdf1']:.1f}% | ≤ 2m {e['cdf2']:.1f}% | ≤ 3m {e['cdf3']:.1f}% | ≤ 5m {e['cdf5']:.1f}%\n"
        "\n"
        "📊 MAN:\n"
        f"  MAE={m['mae']:.3f}m  RMSE={m['rmse']:.3f}m\n"
        f"  P50={m['median']:.3f}m  P90={m['p90']:.3f}m  P95={m['p95']:.3f}m\n"
        f"  Min={m['min']:.3f}m  Max={m['max']:.3f}m\n"
        f"  CDF: ≤ 1m {m['cdf1']:.1f}% | ≤ 2m {m['cdf2']:.1f}% | ≤ 3m {m['cdf3']:.1f}% | ≤ 5m {m['cdf5']:.1f}%\n"
    )

    # noise test
    if not args.no_noise_test:
        print("\n🔊 Noise Robustness Test (EUC + MAN)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        all_x = []
        for x, _ in loader:
            all_x.append(x)
        all_x = torch.cat(all_x, dim=0)  # [N,T,F]
        signal_std = all_x.std().item()

        levels = [float(s.strip()) for s in args.noise_levels.split(",") if s.strip()]

        for lv in levels:
            if args.noise_mode == "percent":
                noise_std = signal_std * (lv / 100.0)
                label = f"{int(lv)}%"
            else:
                noise_std = lv
                label = f"σ={lv}"

            nd_e, nd_m = [], []
            with torch.no_grad():
                for features, targets in loader:
                    features = features.to(device)
                    targets = targets.to(device)

                    noise = torch.randn_like(features) * noise_std
                    noisy = features + noise

                    pred_xy = forward_pred_xy(args.arch, model, noisy, use_amp=use_amp)

                    pred_np = pred_xy.detach().cpu().numpy()
                    tgt_np = targets.detach().cpu().numpy()

                    for i in range(len(pred_np)):
                        pxy = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                        txy = denormalize_coord(tgt_np[i, 0], tgt_np[i, 1])
                        nd_e.append(euclidean_distance_m(pxy, txy))
                        nd_m.append(manhattan_distance_m(pxy, txy))

            noise_mae_e = float(np.mean(nd_e))
            noise_mae_m = float(np.mean(nd_m))
            deg_e = ((noise_mae_e - e["mae"]) / e["mae"]) * 100 if e["mae"] > 1e-9 else 0.0
            deg_m = ((noise_mae_m - m["mae"]) / m["mae"]) * 100 if m["mae"] > 1e-9 else 0.0

            print(
                f"  Noise {label:>6}: "
                f"EUC MAE={noise_mae_e:.3f}m ({deg_e:+.1f}%) | "
                f"MAN MAE={noise_mae_m:.3f}m ({deg_m:+.1f}%)"
            )

    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)

    # ✅ 저장
    if args.save_dir:
        save_results(
            save_dir=Path(args.save_dir),
            arch=args.arch,
            tag=args.tag,
            checkpoint=ckpt_path,
            data_dir=data_dir,
            n_test=len(ds),
            window=window_size,
            n_features=n_features,
            e=e,
            m=m,
            noise_mode=args.noise_mode,
            noise_levels=args.noise_levels,
        )


if __name__ == "__main__":
    main()
