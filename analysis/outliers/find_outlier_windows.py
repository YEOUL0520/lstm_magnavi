#!/usr/bin/env python3
"""33m 아웃라이어 윈도우 찾기 - 어떤 250 타임스텝 경로인지 확인"""
import json
import sys
from pathlib import Path
import torch
import numpy as np

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
        return features, target, idx

def find_outlier_windows(
    checkpoint_path: Path,
    data_dir: Path,
    threshold: float = 5.0,
    device: str = "cpu",
):
    print("=" * 80)
    print("🔍 33m 아웃라이어 윈도우 찾기")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data dir: {data_dir}")
    print()

    # 메타데이터
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

    # Device
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
    outliers = []  # (idx, error, pred, target, features)

    with torch.no_grad():
        for features, targets, indices in test_loader:
            features_cpu = features.clone()
            features = features.to(device)
            targets = targets.to(device)

            edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
            outputs = model(features, edge_ids)
            pred = outputs[:, -1, :]

            pred_np = pred.cpu().numpy()
            target_np = targets.cpu().numpy()
            indices_np = indices.cpu().numpy()
            features_np = features_cpu.numpy()

            for i in range(len(pred_np)):
                pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])

                if dist >= threshold:
                    outliers.append({
                        'idx': int(indices_np[i]),
                        'error': dist,
                        'pred': pred_pos,
                        'target': target_pos,
                        'features': features_np[i]  # (250, 4)
                    })

    print(f"✅ 예측 완료")
    print()

    # 아웃라이어 분석
    outliers.sort(key=lambda x: x['error'], reverse=True)

    print("=" * 80)
    print(f"🎯 아웃라이어 분석 (오차 ≥ {threshold}m)")
    print("=" * 80)
    print(f"  총 샘플: {len(test_ds)}개")
    print(f"  아웃라이어: {len(outliers)}개 ({len(outliers)/len(test_ds)*100:.1f}%)")
    print()

    if len(outliers) == 0:
        print("✅ 아웃라이어 없음!")
        return

    # 상위 10개 출력
    print("📋 상위 10개 아웃라이어 윈도우:")
    print("-" * 80)
    print(f"{'Idx':<8} {'Error(m)':<12} {'Pred(x,y)':<25} {'Target(x,y)':<25}")
    print("-" * 80)

    for i, outlier in enumerate(outliers[:10]):
        idx = outlier['idx']
        error = outlier['error']
        pred = outlier['pred']
        target = outlier['target']

        print(f"{idx:<8} {error:<12.3f} ({pred[0]:>6.2f}, {pred[1]:>6.2f})      ({target[0]:>6.2f}, {target[1]:>6.2f})")

        # 윈도우 특징 분석
        features = outlier['features']  # (250, 4)

        # 시작/끝 좌표 추정 (윈도우 내 이동)
        # features: [MagX, MagY, MagZ, Magnitude] x 250

        print(f"         윈도우 특징:")
        print(f"           MagX: mean={features[:,0].mean():.3f}, std={features[:,0].std():.3f}")
        print(f"           MagY: mean={features[:,1].mean():.3f}, std={features[:,1].std():.3f}")
        print(f"           MagZ: mean={features[:,2].mean():.3f}, std={features[:,2].std():.3f}")
        print(f"           Mag:  mean={features[:,3].mean():.3f}, std={features[:,3].std():.3f}")
        print()

    # 최악의 아웃라이어
    worst = outliers[0]
    print("=" * 80)
    print(f"🔥 최악의 아웃라이어 (오차 {worst['error']:.3f}m)")
    print("=" * 80)
    print(f"  샘플 인덱스: {worst['idx']}")
    print(f"  예측 좌표: ({worst['pred'][0]:.2f}, {worst['pred'][1]:.2f})")
    print(f"  실제 좌표: ({worst['target'][0]:.2f}, {worst['target'][1]:.2f})")
    print(f"  오차: {worst['error']:.3f}m")
    print()

    # 윈도우 250 타임스텝 분석
    features = worst['features']
    print("  250 타임스텝 윈도우 분석:")
    print(f"    MagX 범위: [{features[:,0].min():.3f}, {features[:,0].max():.3f}]")
    print(f"    MagY 범위: [{features[:,1].min():.3f}, {features[:,1].max():.3f}]")
    print(f"    MagZ 범위: [{features[:,2].min():.3f}, {features[:,2].max():.3f}]")
    print(f"    Magnitude 범위: [{features[:,3].min():.3f}, {features[:,3].max():.3f}]")
    print()

    print("  💡 이 윈도우는:")
    print(f"     - Test 데이터의 {worst['idx']}번째 샘플")
    print(f"     - 실제 위치 ({worst['target'][0]:.1f}, {worst['target'][1]:.1f})에 있어야 하지만")
    print(f"     - 모델은 ({worst['pred'][0]:.1f}, {worst['pred'][1]:.1f})로 예측")
    print(f"     - {worst['error']:.1f}m 오차 발생!")
    print()

    print("=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find 33m outlier windows")
    parser.add_argument("--checkpoint", type=str, default="models/hyena_mag4/checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/sliding_mag4")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    find_outlier_windows(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        threshold=args.threshold,
        device=device,
    )
