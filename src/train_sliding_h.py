#!/usr/bin/env python3
"""Sliding Window 방식 학습 - Causal Training"""
import json
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 역정규화
COORD_CENTER = (-44.3, -0.3)
COORD_SCALE = 48.8

def set_seed(seed=42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # 완전한 재현성 (약간의 속도 희생)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WeightedXYLoss(nn.Module):
    """X/Y 방향별 가중치 Loss with Huber (SmoothL1)

    X 방향 오차가 Y보다 3.6배 크므로, X에 더 높은 페널티 적용
    Huber Loss 사용으로 Outlier에 더 강건함
    """
    def __init__(self, x_weight=2.0, y_weight=1.0, delta=1.0):
        super().__init__()
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.huber = nn.SmoothL1Loss(reduction='mean', beta=delta)

    def forward(self, pred, target):
        """
        Args:
            pred: [batch, 2] - (x_norm, y_norm)
            target: [batch, 2] - (x_norm, y_norm)
        """
        x_loss = self.huber(pred[:, 0], target[:, 0]) * self.x_weight
        y_loss = self.huber(pred[:, 1], target[:, 1]) * self.y_weight
        return x_loss + y_loss

def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (x, y)

class SlidingWindowDataset(Dataset):
    """Sliding Window 데이터셋

    각 샘플: {"features": [250, n_features], "target": [x, y]}
    """
    def __init__(self, jsonl_path: Path):
        self.samples = []

        with jsonl_path.open() as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        if self.samples:
            self.n_features = len(self.samples[0]["features"][0])
            self.window_size = len(self.samples[0]["features"])
        else:
            self.n_features = 0
            self.window_size = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features = torch.tensor(sample["features"], dtype=torch.float32)  # [250, n_features]
        target = torch.tensor(sample["target"], dtype=torch.float32)  # [2]

        return features, target

# Hyena 모델 import (기존 것 사용)
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from model_h import HyenaPositioning

def train_sliding(
    data_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    depth: int = 8,
    dropout: float = 0.1,
    patience: int = 10,
    checkpoint_dir: Path = Path("models/hyena_mag4/checkpoints"),
    device: str = "cuda",
    seed: int = 42,
    warmup_epochs: int = 5,
):
    """Sliding Window 학습"""

    # 재현성을 위한 시드 고정
    set_seed(seed)
    print(f"🎲 Random seed: {seed}")

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    meta_path = data_dir / "meta.json"

    # 메타데이터 로드
    with meta_path.open() as f:
        meta = json.load(f)

    n_features = meta["n_features"]
    window_size = meta["window_size"]

    print("=" * 80)
    print("🚀 Sliding Window 학습 시작")
    print("=" * 80)
    print(f"  Features: {n_features}")
    print(f"  Window size: {window_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Depth: {depth}")
    print()

    # Dataset
    train_ds = SlidingWindowDataset(train_path)
    val_ds = SlidingWindowDataset(val_path)
    test_ds = SlidingWindowDataset(test_path)

    print(f"📊 데이터 로드:")
    print(f"  Train: {len(train_ds)}개 샘플")
    print(f"  Val:   {len(val_ds)}개 샘플")
    print(f"  Test:  {len(test_ds)}개 샘플")
    print()

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model - MPS 지원 추가
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 사용 (느림)")

    model = HyenaPositioning(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=2,  # (x, y)
        depth=depth,
        dropout=dropout,
        num_edge_types=1,  # Sliding window에서는 edge 정보 없음
    ).to(device)

    print(f"🧠 모델: Hyena Sliding Window")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Mixed Precision Scaler (CUDA만 지원)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print(f"⚡ Mixed Precision (AMP) 활성화")

    # Learning Rate Scheduler with Warmup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # val_rmse 최소화
        factor=0.5,           # 학습률 절반으로
        patience=5,           # 5 에포크 기다림
        min_lr=1e-6           # 최소 학습률
    )

    # Warmup을 위한 초기 학습률 저장
    base_lr = lr
    warmup_factor = 0.1  # 초기 학습률은 10%부터 시작

    # X 방향 오차가 Y보다 3.6배 크므로, X에 2배 페널티 적용
    criterion = WeightedXYLoss(x_weight=2.0, y_weight=1.0)

    # Training - P90 기준으로 best model 선택
    best_val_p90 = float("inf")
    no_improve = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"

    print("🚀 학습 시작")
    print("   (Best model 기준: P90 - outlier에 강건)\n")

    for epoch in range(1, epochs + 1):
        # Learning Rate Warmup
        if epoch <= warmup_epochs:
            warmup_lr = base_lr * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            if epoch == 1:
                print(f"🔥 Warmup 시작: {warmup_epochs} 에포크 동안 LR {warmup_lr:.2e} → {base_lr:.2e}")

        # Train
        model.train()
        train_loss = 0.0
        train_distances = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=100)
        for features, targets in pbar:
            features = features.to(device)  # [batch, 250, n_features]
            targets = targets.to(device)  # [batch, 2]

            optimizer.zero_grad()

            # Mixed Precision Training
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # Hyena expects [batch, seq_len, features]
                    edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
                    outputs = model(features, edge_ids)  # [batch, 250, 2]
                    pred = outputs[:, -1, :]  # [batch, 2]
                    loss = criterion(pred, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 일반 학습 (CPU)
                edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
                outputs = model(features, edge_ids)
                pred = outputs[:, -1, :]
                loss = criterion(pred, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item() * features.size(0)

            # 거리 계산 (역정규화)
            pred_np = pred.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()

            for i in range(len(pred_np)):
                pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                # Manhattan distance (복도 구조)
                dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])
                train_distances.append(dist)

            # 진행률 표시줄 업데이트
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dist': f'{train_distances[-1]:.2f}m'})

        train_loss /= len(train_ds)
        train_rmse = np.sqrt(np.mean(np.array(train_distances) ** 2))
        train_mae = np.mean(train_distances)

        # Validation
        model.eval()
        val_loss = 0.0
        val_distances = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", ncols=100, leave=False)
            for features, targets in val_pbar:
                features = features.to(device)
                targets = targets.to(device)

                # Validation도 AMP 사용
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
                        outputs = model(features, edge_ids)
                        pred = outputs[:, -1, :]
                        loss = criterion(pred, targets)
                else:
                    edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
                    outputs = model(features, edge_ids)
                    pred = outputs[:, -1, :]
                    loss = criterion(pred, targets)

                val_loss += loss.item() * features.size(0)

                pred_np = pred.cpu().numpy()
                target_np = targets.cpu().numpy()

                for i in range(len(pred_np)):
                    pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                    target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                    # Manhattan distance (복도 구조)
                    dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])
                    val_distances.append(dist)

        val_loss /= len(val_ds)
        val_rmse = np.sqrt(np.mean(np.array(val_distances) ** 2))
        val_mae = np.mean(val_distances)
        val_median = np.median(val_distances)
        val_p90 = np.percentile(val_distances, 90)

        # ReduceLROnPlateau는 warmup 이후에만 작동
        if epoch > warmup_epochs:
            scheduler.step(val_rmse)

        # 현재 학습률 출력
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"[Epoch {epoch:03d}] LR={current_lr:.2e} | "
            f"TrainLoss={train_loss:.4f} TrainRMSE={train_rmse:.3f}m | "
            f"ValRMSE={val_rmse:.3f}m MAE={val_mae:.3f}m "
            f"Median={val_median:.3f}m P90={val_p90:.3f}m"
        )

        # Early stopping - P90 기준
        if val_p90 < best_val_p90 - 0.01:
            best_val_p90 = val_p90
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                    "val_p90": val_p90,
                    "meta": meta,
                },
                best_path,
            )
            print(f"   💾 Best model saved (P90={best_val_p90:.3f}m)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n✅ 학습 완료. Best checkpoint: {best_path}\n")

    # Test - 안전하게 checkpoint 로드
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Checkpoint 로드 성공: {best_path}")
    except Exception as e:
        print(f"⚠️  Checkpoint 로드 실패: {e}")
        print(f"💡 새로 학습한 모델로 Test 평가 진행")
        # 현재 메모리의 모델 사용 (마지막 에포크)

    model.eval()
    test_distances = []

    print("📈 Test 평가 중...")

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", ncols=100)
        for features, targets in test_pbar:
            features = features.to(device)
            targets = targets.to(device)

            # Test도 AMP 사용
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
                # Manhattan distance (복도 구조)
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

    # CDF (Cumulative Distribution Function)
    cdf_1m = np.mean(test_distances_array <= 1.0) * 100
    cdf_2m = np.mean(test_distances_array <= 2.0) * 100
    cdf_3m = np.mean(test_distances_array <= 3.0) * 100
    cdf_5m = np.mean(test_distances_array <= 5.0) * 100

    print(
        f"\n[Test Results]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 기본 메트릭:\n"
        f"  MAE (Mean Absolute):     {test_mae:.3f}m\n"
        f"  RMSE (Root Mean Sq):     {test_rmse:.3f}m\n"
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
    print(f"\n🔊 Noise Robustness Test:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    noise_levels = [0.1, 0.2, 0.5]  # std of Gaussian noise
    for noise_std in noise_levels:
        noise_distances = []

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                targets = targets.to(device)

                # Add Gaussian noise to features
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

        print(f"  Noise σ={noise_std:.1f}: MAE={noise_mae:.3f}m (degradation: {degradation:+.1f}%)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sliding")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="models/hyena_mag4/checkpoints")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Learning rate warmup epochs")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    train_sliding(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        patience=args.patience,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=device,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
    )