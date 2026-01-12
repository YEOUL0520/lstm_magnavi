#!/usr/bin/env python3
"""심층 아웃라이어 분석: Train/Val/Test 모두 분석 + 전처리 일관성 검증"""
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
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
        return features, target, idx  # idx 추가

def analyze_split(model, loader, split_name, device, dataset):
    """한 split의 예측 및 오차 분석"""
    all_errors = []
    all_predictions = []
    all_targets = []
    all_indices = []
    all_features = []

    model.eval()
    with torch.no_grad():
        for features, targets, indices in loader:
            features = features.to(device)
            targets = targets.to(device)

            edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
            outputs = model(features, edge_ids)
            pred = outputs[:, -1, :]

            pred_np = pred.cpu().numpy()
            target_np = targets.cpu().numpy()
            features_np = features.cpu().numpy()

            for i in range(len(pred_np)):
                pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])

                # Manhattan distance
                dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])

                all_errors.append(dist)
                all_predictions.append(pred_pos)
                all_targets.append(target_pos)
                all_indices.append(indices[i].item())
                all_features.append(features_np[i])

    return {
        'errors': np.array(all_errors),
        'predictions': all_predictions,
        'targets': all_targets,
        'indices': all_indices,
        'features': all_features,
        'split': split_name
    }

def check_preprocessing_consistency(data_dir: Path):
    """전처리 일관성 검증"""
    print("=" * 80)
    print("🔍 전처리 일관성 검증")
    print("=" * 80)

    splits = ['train', 'val', 'test']
    stats = {}

    for split in splits:
        jsonl_path = data_dir / f"{split}.jsonl"
        samples = []
        with jsonl_path.open() as f:
            for line in f:
                samples.append(json.loads(line))

        # Feature 통계
        all_features = []
        all_targets = []
        window_sizes = []

        for sample in samples:
            features = np.array(sample['features'])
            target = np.array(sample['target'])

            all_features.append(features)
            all_targets.append(target)
            window_sizes.append(len(features))

        all_features = np.array(all_features)  # [N, 250, n_features]
        all_targets = np.array(all_targets)    # [N, 2]

        # 통계 계산
        stats[split] = {
            'n_samples': len(samples),
            'window_sizes': window_sizes,
            'feature_mean': np.mean(all_features),
            'feature_std': np.std(all_features),
            'feature_min': np.min(all_features),
            'feature_max': np.max(all_features),
            'target_mean': np.mean(all_targets, axis=0),
            'target_std': np.std(all_targets, axis=0),
            'target_min': np.min(all_targets, axis=0),
            'target_max': np.max(all_targets, axis=0),
        }

    # 출력
    print(f"\n{'Split':<10} {'Samples':<10} {'Feature Mean':<15} {'Feature Std':<15} {'Target Mean':<25}")
    print("-" * 80)
    for split in splits:
        s = stats[split]
        print(f"{split:<10} {s['n_samples']:<10} {s['feature_mean']:<15.6f} {s['feature_std']:<15.6f} ({s['target_mean'][0]:>6.3f}, {s['target_mean'][1]:>6.3f})")

    print()
    print("📊 세부 통계:")
    print("-" * 80)
    for split in splits:
        s = stats[split]
        print(f"\n{split.upper()}:")
        print(f"  Window sizes: {set(s['window_sizes'])} (모두 동일해야 함)")
        print(f"  Feature range: [{s['feature_min']:.3f}, {s['feature_max']:.3f}]")
        print(f"  Target range X: [{s['target_min'][0]:.3f}, {s['target_max'][0]:.3f}]")
        print(f"  Target range Y: [{s['target_min'][1]:.3f}, {s['target_max'][1]:.3f}]")

    # 일관성 체크
    print()
    print("=" * 80)
    print("✅ 일관성 체크")
    print("=" * 80)

    # Window size 일관성
    all_window_sizes = [set(stats[s]['window_sizes']) for s in splits]
    if len(set.union(*all_window_sizes)) == 1:
        print("✅ Window size 일관성: OK (모두 동일)")
    else:
        print("❌ Window size 일관성: FAIL (다름)")
        for split in splits:
            print(f"  {split}: {set(stats[split]['window_sizes'])}")

    # Feature 분포 유사성
    feature_means = [stats[s]['feature_mean'] for s in splits]
    feature_stds = [stats[s]['feature_std'] for s in splits]

    mean_diff = max(feature_means) - min(feature_means)
    std_diff = max(feature_stds) - min(feature_stds)

    if mean_diff < 0.1 and std_diff < 0.1:
        print(f"✅ Feature 정규화 일관성: OK (mean diff={mean_diff:.6f}, std diff={std_diff:.6f})")
    else:
        print(f"⚠️ Feature 정규화 차이 있음: mean diff={mean_diff:.6f}, std diff={std_diff:.6f}")

    # Target 분포 유사성
    target_means_x = [stats[s]['target_mean'][0] for s in splits]
    target_means_y = [stats[s]['target_mean'][1] for s in splits]

    mean_diff_x = max(target_means_x) - min(target_means_x)
    mean_diff_y = max(target_means_y) - min(target_means_y)

    if mean_diff_x < 0.1 and mean_diff_y < 0.1:
        print(f"✅ Target 분포 일관성: OK (X diff={mean_diff_x:.6f}, Y diff={mean_diff_y:.6f})")
    else:
        print(f"⚠️ Target 분포 차이 있음: X diff={mean_diff_x:.6f}, Y diff={mean_diff_y:.6f}")

    print()
    return stats

def deep_analyze_outliers(
    checkpoint_path: Path,
    data_dir: Path,
    threshold: float = 5.0,
    device: str = "cpu",
):
    """심층 아웃라이어 분석: Train/Val/Test 전부"""

    print("=" * 80)
    print("🔬 심층 아웃라이어 분석 (Train/Val/Test)")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data dir: {data_dir}")
    print(f"  Outlier threshold: {threshold}m")
    print()

    # 1. 전처리 일관성 검증
    preprocessing_stats = check_preprocessing_consistency(data_dir)

    # 메타데이터 로드
    meta_path = data_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)

    n_features = meta["n_features"]

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
    print()
    print("🔄 Checkpoint 로드 중...")
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

    # 2. 각 split 분석
    splits = ['train', 'val', 'test']
    results = {}

    for split in splits:
        print(f"📊 {split.upper()} 분석 중...")
        jsonl_path = data_dir / f"{split}.jsonl"
        dataset = SlidingWindowDataset(jsonl_path)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        results[split] = analyze_split(model, loader, split, device, dataset)
        print(f"✅ {split.upper()} 완료: {len(results[split]['errors'])}개 샘플")

    print()

    # 3. Split별 통계 비교
    print("=" * 80)
    print("📊 Split별 성능 비교")
    print("=" * 80)
    print(f"{'Split':<10} {'Samples':<10} {'Mean':<10} {'Median':<10} {'P90':<10} {'P95':<10} {'Max':<10} {'Outliers':<12}")
    print("-" * 80)

    for split in splits:
        errors = results[split]['errors']
        outliers = np.sum(errors >= threshold)
        outlier_pct = outliers / len(errors) * 100

        print(f"{split:<10} {len(errors):<10} {np.mean(errors):<10.3f} {np.median(errors):<10.3f} "
              f"{np.percentile(errors, 90):<10.3f} {np.percentile(errors, 95):<10.3f} "
              f"{np.max(errors):<10.3f} {outliers:<5}({outlier_pct:>4.1f}%)")

    print()

    # 4. 아웃라이어 상세 분석
    print("=" * 80)
    print(f"🎯 아웃라이어 상세 분석 (≥{threshold}m)")
    print("=" * 80)

    for split in splits:
        errors = results[split]['errors']
        outlier_indices = np.where(errors >= threshold)[0]

        if len(outlier_indices) == 0:
            print(f"\n{split.upper()}: ✅ 아웃라이어 없음")
            continue

        print(f"\n{split.upper()}: {len(outlier_indices)}개 아웃라이어")
        print("-" * 80)
        print(f"{'Index':<8} {'Error(m)':<12} {'Pred(x,y)':<25} {'Target(x,y)':<25}")
        print("-" * 80)

        # 오차 큰 순으로 정렬
        sorted_outliers = outlier_indices[np.argsort(errors[outlier_indices])[::-1]]

        for idx in sorted_outliers[:10]:  # 상위 10개
            error = errors[idx]
            pred = results[split]['predictions'][idx]
            target = results[split]['targets'][idx]
            print(f"{idx:<8} {error:<12.3f} ({pred[0]:>6.2f}, {pred[1]:>6.2f})      "
                  f"({target[0]:>6.2f}, {target[1]:>6.2f})")

        if len(sorted_outliers) > 10:
            print(f"... ({len(sorted_outliers)}개 중 10개만 표시)")

    print()

    # 5. Feature 분석 (아웃라이어 vs 정상)
    print("=" * 80)
    print("📈 Feature 분석 (아웃라이어 vs 정상)")
    print("=" * 80)

    for split in splits:
        errors = results[split]['errors']
        features = np.array(results[split]['features'])  # [N, 250, n_features]

        outlier_mask = errors >= threshold
        normal_mask = errors < threshold

        if np.sum(outlier_mask) == 0:
            print(f"\n{split.upper()}: 아웃라이어 없음")
            continue

        outlier_features = features[outlier_mask]
        normal_features = features[normal_mask]

        print(f"\n{split.upper()}:")
        print(f"  정상 샘플 feature 평균: {np.mean(normal_features):.6f} (std: {np.std(normal_features):.6f})")
        print(f"  아웃라이어 feature 평균: {np.mean(outlier_features):.6f} (std: {np.std(outlier_features):.6f})")
        print(f"  차이: {abs(np.mean(outlier_features) - np.mean(normal_features)):.6f}")

        # 각 feature 차원별로
        for feat_idx in range(n_features):
            normal_feat = normal_features[:, :, feat_idx]
            outlier_feat = outlier_features[:, :, feat_idx]

            feat_names = ['MagX', 'MagY', 'MagZ', 'Magnitude']
            print(f"  Feature {feat_idx} ({feat_names[feat_idx] if feat_idx < len(feat_names) else feat_idx}):")
            print(f"    정상: mean={np.mean(normal_feat):.6f}, std={np.std(normal_feat):.6f}")
            print(f"    아웃: mean={np.mean(outlier_feat):.6f}, std={np.std(outlier_feat):.6f}")

    print()

    # 6. 시각화
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 6-1. Split별 오차 분포 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, split in enumerate(splits):
        errors = results[split]['errors']

        # 히스토그램
        ax = axes[0, idx]
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(threshold, color='red', linestyle='--', label=f'{threshold}m')
        ax.set_xlabel('Error (m)', fontproperties=font_prop)
        ax.set_ylabel('Count', fontproperties=font_prop)
        ax.set_title(f'{split.upper()} 오차 분포', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        ax.grid(True, alpha=0.3)

        # CDF
        ax = axes[1, idx]
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cdf, linewidth=2)
        ax.axvline(threshold, color='red', linestyle='--')
        ax.axhline(90, color='orange', linestyle='--', label='P90')
        ax.set_xlabel('Error (m)', fontproperties=font_prop)
        ax.set_ylabel('Cumulative %', fontproperties=font_prop)
        ax.set_title(f'{split.upper()} CDF', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(10, np.max(errors)))

    plt.tight_layout()
    output_path = output_dir / "deep_outlier_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 시각화 저장: {output_path}")

    # 6-2. Split 비교 박스플롯
    plt.figure(figsize=(12, 6))

    all_errors = [results[s]['errors'] for s in splits]
    plt.boxplot(all_errors, labels=[s.upper() for s in splits])
    plt.axhline(threshold, color='red', linestyle='--', label=f'Outlier threshold ({threshold}m)')
    plt.ylabel('Error (m)', fontproperties=font_prop)
    plt.title('Split별 오차 분포 비교', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3, axis='y')

    output_path2 = output_dir / "split_comparison.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"📊 비교 시각화 저장: {output_path2}")

    print()
    print("=" * 80)
    print("✅ 심층 분석 완료!")
    print("=" * 80)

    # 7. 결론
    print()
    print("=" * 80)
    print("📋 결론")
    print("=" * 80)

    # 과적합 여부
    train_rmse = np.sqrt(np.mean(results['train']['errors']**2))
    val_rmse = np.sqrt(np.mean(results['val']['errors']**2))
    test_rmse = np.sqrt(np.mean(results['test']['errors']**2))

    print(f"  Train RMSE: {train_rmse:.3f}m")
    print(f"  Val RMSE:   {val_rmse:.3f}m")
    print(f"  Test RMSE:  {test_rmse:.3f}m")
    print()

    if abs(val_rmse - test_rmse) < 0.3:
        print("✅ Val/Test 성능 유사 → 과적합 없음")
    else:
        print(f"⚠️ Val/Test 차이 있음: {abs(val_rmse - test_rmse):.3f}m")

    if train_rmse < val_rmse < val_rmse + 1.0:
        print("✅ Train/Val 차이 정상 범위")
    elif train_rmse > val_rmse:
        print("⚠️ Train이 Val보다 나쁨 → 학습 부족?")
    else:
        print(f"⚠️ Train/Val 차이 큼: {val_rmse - train_rmse:.3f}m → 과적합 가능성")

    print()

    # 아웃라이어 분포
    for split in splits:
        outliers = np.sum(results[split]['errors'] >= threshold)
        total = len(results[split]['errors'])
        pct = outliers / total * 100
        if pct > 5:
            print(f"⚠️ {split.upper()}: 아웃라이어 {pct:.1f}% ({outliers}/{total}) - 많음")
        else:
            print(f"✅ {split.upper()}: 아웃라이어 {pct:.1f}% ({outliers}/{total})")

    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep outlier analysis across all splits")
    parser.add_argument("--checkpoint", type=str, default="models/hyena_mag4/checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/sliding_mag4")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    deep_analyze_outliers(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        threshold=args.threshold,
        device=device,
    )
