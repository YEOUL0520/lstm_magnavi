#!/usr/bin/env python3
"""
Outlier 분석 및 필터링 파이프라인

사용법:
1. 학습 완료 후 실행:
   python scripts/analyze_and_filter_outliers.py --checkpoint new/models/hyena_mag4/checkpoints/best.pt

2. Outlier 분석 결과 확인

3. 원하는 threshold로 데이터 필터링:
   python scripts/analyze_and_filter_outliers.py --checkpoint ... --filter --threshold 5.0

4. 필터링된 데이터로 재학습
"""
import torch
import json
import numpy as np
from pathlib import Path
import sys
import argparse

# 모델 import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from model import HyenaModel

COORD_CENTER = (-44.3, -0.3)
COORD_SCALE = 48.8

def denormalize_coord(coord_norm):
    """정규화된 좌표를 실제 좌표로 변환"""
    x = coord_norm[0] * COORD_SCALE + COORD_CENTER[0]
    y = coord_norm[1] * COORD_SCALE + COORD_CENTER[1]
    return x, y

def load_data(data_path):
    """데이터 로드"""
    samples = []
    with data_path.open() as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def analyze_outliers(checkpoint_path, data_dir, hidden_dim=384, depth=10):
    """
    모델로 예측해서 오차 큰 샘플 찾기

    Returns:
        dict: {
            'errors': [오차 리스트],
            'predictions': [예측 정보],
            'outlier_indices': {threshold: [인덱스]}
        }
    """
    print("=" * 80)
    print("Outlier 분석")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 모델 로드
    print(f"\n모델 로드: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Feature 수 추정 (meta.json에서)
    meta_path = data_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)
    n_features = meta.get("n_features", 4)

    model = HyenaModel(
        in_dim=n_features,
        hidden_dim=hidden_dim,
        out_dim=2,
        depth=depth,
        order=2
    ).to(device)

    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 모델 로드 완료 (features={n_features}, hidden={hidden_dim}, depth={depth})")

    # 데이터 로드
    results = {}

    for split in ["train", "val", "test"]:
        data_path = data_dir / f"{split}.jsonl"
        if not data_path.exists():
            continue

        print(f"\n{split.upper()} 데이터 분석 중...")
        samples = load_data(data_path)

        errors = []
        predictions = []

        with torch.no_grad():
            for i, sample in enumerate(samples):
                # Input
                features = torch.tensor(sample["features"], dtype=torch.float32).unsqueeze(0).to(device)
                target = torch.tensor(sample["target"], dtype=torch.float32).to(device)

                # Predict
                pred = model(features).squeeze(0)

                # Denormalize
                pred_x, pred_y = denormalize_coord(pred.cpu().numpy())
                true_x, true_y = denormalize_coord(target.cpu().numpy())

                # Euclidean distance
                error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)

                errors.append(error)
                predictions.append({
                    'index': i,
                    'error': error,
                    'pred': (pred_x, pred_y),
                    'true': (true_x, true_y),
                    'features': sample["features"],
                    'target': sample["target"]
                })

        errors = np.array(errors)

        # 통계
        print(f"  샘플: {len(errors)}개")
        print(f"  Mean:   {np.mean(errors):.3f}m")
        print(f"  Median: {np.median(errors):.3f}m")
        print(f"  P90:    {np.percentile(errors, 90):.3f}m")
        print(f"  P95:    {np.percentile(errors, 95):.3f}m")
        print(f"  Max:    {np.max(errors):.3f}m")

        # Outlier threshold별 개수
        print(f"\n  Outlier 분포:")
        for thresh in [3, 4, 5, 6, 7, 8, 10]:
            count = np.sum(errors > thresh)
            pct = count * 100 / len(errors)
            print(f"    > {thresh}m: {count:4d}개 ({pct:5.1f}%)")

        # Outlier 인덱스 저장
        outlier_indices = {}
        for thresh in [3, 4, 5, 6, 7, 8]:
            outlier_indices[thresh] = [p['index'] for p in predictions if p['error'] > thresh]

        results[split] = {
            'errors': errors.tolist(),
            'predictions': predictions,
            'outlier_indices': outlier_indices,
            'stats': {
                'mean': float(np.mean(errors)),
                'median': float(np.median(errors)),
                'std': float(np.std(errors)),
                'p90': float(np.percentile(errors, 90)),
                'p95': float(np.percentile(errors, 95)),
                'p99': float(np.percentile(errors, 99)),
                'max': float(np.max(errors))
            }
        }

    return results

def filter_data(data_dir, outlier_results, threshold, output_dir):
    """
    Outlier 제거한 데이터 생성

    Args:
        data_dir: 원본 데이터 디렉토리
        outlier_results: analyze_outliers 결과
        threshold: 오차 threshold (m)
        output_dir: 필터링된 데이터 저장 디렉토리
    """
    print("\n" + "=" * 80)
    print(f"데이터 필터링 (threshold={threshold}m)")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    total_kept = 0

    for split in ["train", "val", "test"]:
        if split not in outlier_results:
            continue

        # 원본 데이터 로드
        data_path = data_dir / f"{split}.jsonl"
        samples = load_data(data_path)

        # Outlier 인덱스
        outlier_indices = set(outlier_results[split]['outlier_indices'].get(threshold, []))

        # 필터링
        filtered_samples = []
        for i, sample in enumerate(samples):
            if i not in outlier_indices:
                filtered_samples.append(sample)

        # 저장
        output_path = output_dir / f"{split}.jsonl"
        with output_path.open('w') as f:
            for sample in filtered_samples:
                f.write(json.dumps(sample) + '\n')

        removed = len(samples) - len(filtered_samples)
        total_removed += removed
        total_kept += len(filtered_samples)

        print(f"\n{split.upper()}:")
        print(f"  원본:     {len(samples):5d}개")
        print(f"  제거:     {removed:5d}개 ({removed*100/len(samples):.1f}%)")
        print(f"  필터링됨: {len(filtered_samples):5d}개")
        print(f"  저장:     {output_path}")

    # 메타데이터 복사 및 업데이트
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)

        # 샘플 수 업데이트
        for split in ["train", "val", "test"]:
            if split in outlier_results:
                key = f"n_{split}"
                original = meta.get(key, 0)
                removed = len(outlier_results[split]['outlier_indices'].get(threshold, []))
                meta[key] = original - removed

        meta['filtered'] = True
        meta['filter_threshold'] = threshold

        output_meta_path = output_dir / "meta.json"
        with output_meta_path.open('w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n메타데이터 저장: {output_meta_path}")

    print(f"\n총 제거: {total_removed}개")
    print(f"총 유지: {total_kept}개")
    print(f"\n✅ 필터링 완료: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Outlier 분석 및 필터링")
    parser.add_argument("--checkpoint", required=True, help="모델 checkpoint 경로")
    parser.add_argument("--data-dir", default="new/data/sliding_mag4", help="데이터 디렉토리")
    parser.add_argument("--hidden-dim", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=10, help="Model depth")
    parser.add_argument("--filter", action="store_true", help="Outlier 제거된 데이터 생성")
    parser.add_argument("--threshold", type=float, default=5.0, help="Outlier threshold (m)")
    parser.add_argument("--output-dir", default="new/data/sliding_mag4_filtered", help="필터링된 데이터 저장 경로")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)

    # Outlier 분석
    results = analyze_outliers(checkpoint_path, data_dir, args.hidden_dim, args.depth)

    # 결과 저장
    output_file = Path("analysis/outputs/outlier_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # predictions는 저장 안 함 (너무 큼)
    save_results = {}
    for split, data in results.items():
        save_results[split] = {
            'stats': data['stats'],
            'outlier_indices': data['outlier_indices'],
            'n_samples': len(data['errors'])
        }

    with output_file.open('w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n분석 결과 저장: {output_file}")

    # 필터링 수행
    if args.filter:
        filter_data(data_dir, results, args.threshold, args.output_dir)

        print("\n" + "=" * 80)
        print("다음 단계:")
        print("=" * 80)
        print(f"1. 필터링된 데이터로 재학습:")
        print(f"   python new/src/train_sliding.py \\")
        print(f"     --data-dir {args.output_dir} \\")
        print(f"     --epochs 100 \\")
        print(f"     --batch-size 128 \\")
        print(f"     --hidden-dim {args.hidden_dim} \\")
        print(f"     --depth {args.depth}")

if __name__ == "__main__":
    main()
