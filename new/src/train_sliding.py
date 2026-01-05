#!/usr/bin/env python3
"""Sliding Window 방식 학습 - 깔끔한 MSE Loss 버전"""
import json
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 역정규화 (현재 버전 유지)
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

# Hyena 모델 import
import sys
sys.path.append(str(Path(__file__).parent))
from model import HyenaPositioning

def train_sliding(
    data_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    depth: int = 8,
    dropout: float = 0.1,
    patience: int = 10,
    checkpoint_dir: Path = Path("new/models/hyena_mag4/checkpoints"),
    device: str = "cuda",
    seed: int = 42,
    warmup_epochs: int = 5,
):
    """Sliding Window 학습 - 깔끔한 MSE Loss 버전"""

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
    print("🚀 Sliding Window 학습 시작 (Clean MSE Version)")
    print("=" * 80)
    print(f"  Features: {n_features}")
    print(f"  Window size: {window_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Depth: {depth}")
    print(f"  정규화: COORD_CENTER={COORD_CENTER}, COORD_SCALE={COORD_SCALE}")
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

    # Loss function - 깔끔한 MSE Loss
    criterion = nn.MSELoss()
    print(f"📉 Loss Function: MSELoss (No weights)")

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
                # 일반 학습 (MPS/CPU)
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
                # Euclidean distance
                dist = math.sqrt((pred_pos[0] - target_pos[0])**2 + (pred_pos[1] - target_pos[1])**2)
                train_distances.append(dist)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dist': f'{train_distances[-1]:.2f}m'})

        train_loss /= len(train_ds)
        train_rmse = np.sqrt(np.mean(np.array(train_distances) ** 2))
        train_mae = np.mean(train_distances)

        # Validation
        model.eval()
        val_loss = 0.0
        val_distances = []

        with torch.no_grad():
            for features, targets in val_loader:
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

                # 거리 계산
                pred_np = pred.cpu().numpy()
                target_np = targets.cpu().numpy()

                for i in range(len(pred_np)):
                    pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                    target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                    dist = math.sqrt((pred_pos[0] - target_pos[0])**2 + (pred_pos[1] - target_pos[1])**2)
                    val_distances.append(dist)

        val_loss /= len(val_ds)
        val_distances_array = np.array(val_distances)
        val_rmse = np.sqrt(np.mean(val_distances_array ** 2))
        val_mae = np.mean(val_distances_array)
        val_p90 = np.percentile(val_distances_array, 90)

        # Learning Rate Scheduling (Warmup 이후)
        if epoch > warmup_epochs:
            scheduler.step(val_rmse)

        # Best model 저장 (P90 기준)
        if val_p90 < best_val_p90:
            best_val_p90 = val_p90
            no_improve = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_p90': val_p90,
                'config': {
                    'input_dim': n_features,
                    'hidden_dim': hidden_dim,
                    'output_dim': 2,
                    'depth': depth,
                    'dropout': dropout,
                    'window_size': window_size,
                    'coord_center': COORD_CENTER,
                    'coord_scale': COORD_SCALE,
                }
            }, best_path)

            print(f"  ✅ Best model 저장! (P90={val_p90:.3f}m, RMSE={val_rmse:.3f}m)")
        else:
            no_improve += 1

        # Epoch 결과 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"  Epoch {epoch:3d}: "
            f"Train Loss={train_loss:.4f} (MAE={train_mae:.3f}m) | "
            f"Val Loss={val_loss:.4f} (MAE={val_mae:.3f}m, P90={val_p90:.3f}m) | "
            f"LR={current_lr:.2e}"
        )

        # Early Stopping
        if no_improve >= patience:
            print(f"\n⏹️  Early stopping (no improve for {patience} epochs)")
            break

    # Test evaluation
    print("\n" + "=" * 80)
    print("📊 Test 평가 (Best Model)")
    print("=" * 80)

    # Best 모델 로드
    checkpoint = torch.load(best_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_distances = []

    with torch.no_grad():
        for features, targets in test_loader:
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
                dist = math.sqrt((pred_pos[0] - target_pos[0])**2 + (pred_pos[1] - target_pos[1])**2)
                test_distances.append(dist)

    test_distances_array = np.array(test_distances)

    # 기본 메트릭
    test_rmse = np.sqrt(np.mean(test_distances_array ** 2))
    test_mae = np.mean(test_distances_array)

    # Percentiles
    test_median = np.median(test_distances_array)
    test_p90 = np.percentile(test_distances_array, 90)
    test_p95 = np.percentile(test_distances_array, 95)

    # CDF
    within_1m = (test_distances_array <= 1.0).sum() / len(test_distances_array) * 100
    within_2m = (test_distances_array <= 2.0).sum() / len(test_distances_array) * 100
    within_3m = (test_distances_array <= 3.0).sum() / len(test_distances_array) * 100

    print(
        f"\n[Test Results]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 기본 메트릭:\n"
        f"  MAE (Mean Absolute):     {test_mae:.3f}m\n"
        f"  RMSE (Root Mean Sq):     {test_rmse:.3f}m\n"
        f"\n"
        f"📈 분포:\n"
        f"  Median (P50): {test_median:.3f}m\n"
        f"  P90:          {test_p90:.3f}m\n"
        f"  P95:          {test_p95:.3f}m\n"
        f"\n"
        f"📍 CDF (누적 분포):\n"
        f"  ≤ 1m:  {within_1m:.1f}%\n"
        f"  ≤ 2m:  {within_2m:.1f}%\n"
        f"  ≤ 3m:  {within_3m:.1f}%\n"
    )

    print("=" * 80)
    print(f"✅ 학습 완료!")
    print(f"   Best Model: {best_path}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Sliding window JSONL 디렉토리")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="new/models/hyena_mag4/checkpoints")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Learning rate warmup epochs")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

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
