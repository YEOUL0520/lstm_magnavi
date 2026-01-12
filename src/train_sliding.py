#!/usr/bin/env python3
"""
Sliding Window 방식 학습 - LSTM 전용 (Hyena 제거)
+ OOM 발생 시 batch 자동 감소 후 재시도

- 입력: data_dir/{train,val,test}.jsonl + meta.json
- 출력:
  - 학습/검증 중 EUC + MAN 거리 오차를 동시에 출력
  - best checkpoint는 "Val EUC P90" 기준 저장

[Auto batch retry]
- CUDA OOM이 발생하면 batch_size를 절반으로 낮추고(최소 min_batch_size),
  해당 epoch를 처음부터 다시 실행합니다.
- batch_size가 바뀌면 이후 epoch에서도 새 batch_size로 계속 학습합니다.
"""

import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from hyena.src.model_h import LSTMPositioning

COORD_CENTER = (-44.3, -0.3)
COORD_SCALE = 48.8


# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (float(x), float(y))


def manhattan_distance_m(pred_xy, true_xy) -> float:
    return abs(pred_xy[0] - true_xy[0]) + abs(pred_xy[1] - true_xy[1])


def euclidean_distance_m(pred_xy, true_xy) -> float:
    dx = pred_xy[0] - true_xy[0]
    dy = pred_xy[1] - true_xy[1]
    return float(np.sqrt(dx * dx + dy * dy))


def summarize_dist(dist: np.ndarray) -> dict:
    dist = dist.astype(np.float32)
    return {
        "mae": float(np.mean(dist)),
        "rmse": float(np.sqrt(np.mean(dist ** 2))),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "min": float(np.min(dist)),
        "max": float(np.max(dist)),
    }


def is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        isinstance(err, RuntimeError)
        and ("out of memory" in msg or "cuda" in msg and "memory" in msg)
    )


def cleanup_after_oom(device: torch.device):
    # CUDA 캐시 해제 + 동기화
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# -------------------------
# Dataset
# -------------------------
class SlidingWindowDataset(Dataset):
    """각 샘플: {"features": [T, F], "target": [2]}"""

    def __init__(self, jsonl_path: Path):
        self.samples = []
        with jsonl_path.open() as f:
            for line in f:
                self.samples.append(json.loads(line))

        if self.samples:
            self.n_features = len(self.samples[0]["features"][0])
            self.window_size = len(self.samples[0]["features"])
        else:
            self.n_features = 0
            self.window_size = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["features"], dtype=torch.float32)  # [T, F]
        y = torch.tensor(s["target"], dtype=torch.float32)    # [2]
        return x, y


# -------------------------
# One epoch (train/val)
# -------------------------
def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float,
    epoch: int,
    epochs: int,
):
    model.train()
    total_loss = 0.0
    dist_euc, dist_man = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=110)
    for features, targets in pbar:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                pred = model(features)
                loss = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(features)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * features.size(0)

        pred_np = pred.detach().cpu().numpy()
        tgt_np = targets.detach().cpu().numpy()
        for i in range(len(pred_np)):
            pxy = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
            txy = denormalize_coord(tgt_np[i, 0], tgt_np[i, 1])
            dist_euc.append(euclidean_distance_m(pxy, txy))
            dist_man.append(manhattan_distance_m(pxy, txy))

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "euc(m)": f"{dist_euc[-1]:.2f}",
            "man(m)": f"{dist_man[-1]:.2f}",
        })

    total_loss /= max(1, len(loader.dataset))
    return total_loss, summarize_dist(np.array(dist_euc)), summarize_dist(np.array(dist_man))


def run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int,
):
    model.eval()
    total_loss = 0.0
    dist_euc, dist_man = [], []

    with torch.no_grad():
        vbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", ncols=110, leave=False)
        for features, targets in vbar:
            features = features.to(device)
            targets = targets.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(features)
                    loss = criterion(pred, targets)
            else:
                pred = model(features)
                loss = criterion(pred, targets)

            total_loss += loss.item() * features.size(0)

            pred_np = pred.detach().cpu().numpy()
            tgt_np = targets.detach().cpu().numpy()
            for i in range(len(pred_np)):
                pxy = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                txy = denormalize_coord(tgt_np[i, 0], tgt_np[i, 1])
                dist_euc.append(euclidean_distance_m(pxy, txy))
                dist_man.append(manhattan_distance_m(pxy, txy))

    total_loss /= max(1, len(loader.dataset))
    return total_loss, summarize_dist(np.array(dist_euc)), summarize_dist(np.array(dist_man))


# -------------------------
# Train (with auto batch retry)
# -------------------------
def train(
    data_dir: Path,
    epochs: int = 500,
    batch_size: int = 300,
    min_batch_size: int = 16,
    lr: float = 5e-4,
    hidden_dim: int = 600,
    num_layers: int = 3,
    dropout: float = 0.0,
    patience: int = 15,
    checkpoint_dir: Path = Path("checkpoints"),
    device: str = "cuda",
    seed: int = 42,
    grad_clip: float = 1.0,
    use_fc_relu: bool = False,
    num_workers: int = 0,
):
    set_seed(seed)
    print(f"🎲 Random seed: {seed}")

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    meta_path = data_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    with meta_path.open() as f:
        meta = json.load(f)

    n_features = int(meta["n_features"])
    window_size = int(meta["window_size"])

    print("=" * 80)
    print("🚀 LSTM Sliding Window Training (EUC + MAN) + AutoBatchRetry")
    print("=" * 80)
    print(f"  Data dir:     {data_dir}")
    print(f"  Features(F):  {n_features}")
    print(f"  Window(T):    {window_size}")
    print(f"  LSTM: layers={num_layers}, hidden={hidden_dim}, dropout={dropout}, fc_relu={use_fc_relu}")
    print(f"  epochs={epochs}, batch(start)={batch_size}, batch(min)={min_batch_size}, lr={lr}")
    print("  metrics: euclidean + manhattan (both)")
    print()

    train_ds = SlidingWindowDataset(train_path)
    val_ds = SlidingWindowDataset(val_path)
    test_ds = SlidingWindowDataset(test_path)

    print("📊 데이터 로드:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")
    print()

    # Device
    if device == "cuda" and torch.cuda.is_available():
        device_t = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_t = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) 사용")
    else:
        device_t = torch.device("cpu")
        print("⚠️  CPU 사용")

    # Model
    model = LSTMPositioning(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_fc_relu=use_fc_relu,
    ).to(device_t)

    print("🧠 모델: LSTMPositioning")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 논문 참고: Adam + MSE + lr=5e-4
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # AMP (CUDA only)
    use_amp = device_t.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("⚡ Mixed Precision (AMP) 활성화")

    # Checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"

    # Best 기준: Val EUC P90
    best_val_p90_euc = float("inf")
    no_improve = 0

    # DataLoader builder (batch_size 변경될 수 있음)
    def make_loaders(bs: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )
        return train_loader, val_loader, test_loader

    current_bs = int(batch_size)
    train_loader, val_loader, test_loader = make_loaders(current_bs)

    print("🚀 학습 시작")
    print("   (Best 기준: Val EUC P90)")
    print("   (OOM 발생 시 batch를 절반으로 낮추고 해당 epoch 재시도)\n")

    epoch = 1
    while epoch <= epochs:
        # epoch 단위 재시도 루프
        while True:
            try:
                start_t = time.time()

                train_loss, train_e, train_m = run_train_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device_t,
                    use_amp=use_amp,
                    scaler=scaler,
                    grad_clip=grad_clip,
                    epoch=epoch,
                    epochs=epochs,
                )

                val_loss, val_e, val_m = run_val_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device_t,
                    use_amp=use_amp,
                    epoch=epoch,
                    epochs=epochs,
                )

                elapsed = time.time() - start_t

                print(
                    f"[Epoch {epoch:03d}] (bs={current_bs}) {elapsed:.1f}s | "
                    f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}\n"
                    f"   Train  EUC: MAE={train_e['mae']:.3f} RMSE={train_e['rmse']:.3f} | "
                    f"MAN: MAE={train_m['mae']:.3f} RMSE={train_m['rmse']:.3f}\n"
                    f"   Val    EUC: MAE={val_e['mae']:.3f} RMSE={val_e['rmse']:.3f} "
                    f"Med={val_e['median']:.3f} P90={val_e['p90']:.3f} P95={val_e['p95']:.3f} | "
                    f"MAN: MAE={val_m['mae']:.3f} RMSE={val_m['rmse']:.3f} "
                    f"Med={val_m['median']:.3f} P90={val_m['p90']:.3f} P95={val_m['p95']:.3f}"
                )

                # Best 저장: Val EUC P90
                if val_e["p90"] < best_val_p90_euc - 0.01:
                    best_val_p90_euc = val_e["p90"]
                    no_improve = 0

                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch,
                            "val_metrics": {"euc": val_e, "man": val_m},
                            "meta": meta,
                            "config": {
                                "hidden_dim": hidden_dim,
                                "num_layers": num_layers,
                                "dropout": dropout,
                                "use_fc_relu": use_fc_relu,
                                "batch_size": current_bs,
                            },
                        },
                        best_path,
                    )
                    print(f"   💾 Best model saved (Val EUC P90={best_val_p90_euc:.3f}m) -> {best_path}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"\n⏹️  Early stopping at epoch {epoch} (no improve {patience} epochs)")
                        epoch = epochs + 1  # outer loop 탈출
                        break

                # epoch 성공 -> retry 루프 탈출
                break

            except RuntimeError as e:
                if is_cuda_oom(e) and device_t.type == "cuda":
                    print("\n🔥 CUDA OOM 발생! batch를 낮춰서 재시도합니다.")
                    print(f"   현재 batch_size={current_bs}")

                    cleanup_after_oom(device_t)

                    new_bs = current_bs // 2
                    if new_bs < min_batch_size:
                        print(f"❌ batch를 더 낮출 수 없음 (min_batch_size={min_batch_size}). 종료합니다.")
                        raise

                    current_bs = new_bs
                    print(f"   → 새 batch_size={current_bs} 로 epoch {epoch} 재시도\n")

                    # loader 재생성
                    train_loader, val_loader, test_loader = make_loaders(current_bs)
                    continue
                else:
                    raise

        epoch += 1

    print(f"\n✅ 학습 종료. Best checkpoint: {best_path}\n")

    # --------------------
    # Test (best checkpoint)
    # --------------------
    checkpoint = torch.load(best_path, map_location=device_t, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_dist_euc = []
    test_dist_man = []

    with torch.no_grad():
        tbar = tqdm(test_loader, desc="Testing (Best)", ncols=110)
        for features, targets in tbar:
            features = features.to(device_t)
            targets = targets.to(device_t)

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
                test_dist_euc.append(euclidean_distance_m(pxy, txy))
                test_dist_man.append(manhattan_distance_m(pxy, txy))

    test_e = summarize_dist(np.array(test_dist_euc))
    test_m = summarize_dist(np.array(test_dist_man))

    print(
        "\n[Test Results - Best]\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"EUC  MAE={test_e['mae']:.3f}  RMSE={test_e['rmse']:.3f}  "
        f"P50={test_e['median']:.3f}  P90={test_e['p90']:.3f}  P95={test_e['p95']:.3f}\n"
        f"MAN  MAE={test_m['mae']:.3f}  RMSE={test_m['rmse']:.3f}  "
        f"P50={test_m['median']:.3f}  P90={test_m['p90']:.3f}  P95={test_m['p95']:.3f}\n"
    )

    return best_path


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/sliding_lstm")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--min-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=600)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-fc-relu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    train(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_batch_size=args.min_batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=device,
        seed=args.seed,
        grad_clip=args.grad_clip,
        use_fc_relu=args.use_fc_relu,
        num_workers=args.num_workers,
    )