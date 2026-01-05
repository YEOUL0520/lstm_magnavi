#!/usr/bin/env python3
"""극단적 Outlier (10m 이상) 상세 분석"""
import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.path.append('src')

from model import Hyena
from train_sliding import denormalize_coord

# 설정
CHECKPOINT = Path("models/hyena_mag4/checkpoints/best.pt")
DATA_DIR = Path("data/sliding_mag4")
EXTREME_THRESHOLD = 10.0  # 10m 이상을 극단적 outlier로 정의
OUTPUT_FILE = Path("analysis/outputs/extreme_outliers.txt")

def load_model(checkpoint_path, device):
    """모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = Hyena(
        n_features=checkpoint['n_features'],
        hidden_dim=checkpoint['hidden_dim'],
        depth=checkpoint['depth'],
        dropout=checkpoint.get('dropout', 0.1)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

def analyze_sample_features(features):
    """샘플의 feature 통계 분석"""
    features = np.array(features)  # [250, 6]

    # MagX, MagY, MagZ (정규화된 값)
    magx = features[:, 0]
    magy = features[:, 1]
    magz = features[:, 2]

    stats = {
        'magx_mean': float(np.mean(magx)),
        'magx_std': float(np.std(magx)),
        'magx_range': float(np.max(magx) - np.min(magx)),
        'magy_mean': float(np.mean(magy)),
        'magy_std': float(np.std(magy)),
        'magy_range': float(np.max(magy) - np.min(magy)),
        'magz_mean': float(np.mean(magz)),
        'magz_std': float(np.std(magz)),
        'magz_range': float(np.max(magz) - np.min(magz)),
    }

    # 변화율 (얼마나 급변하는지)
    stats['magx_change_rate'] = float(np.mean(np.abs(np.diff(magx))))
    stats['magy_change_rate'] = float(np.mean(np.abs(np.diff(magy))))
    stats['magz_change_rate'] = float(np.mean(np.abs(np.diff(magz))))

    return stats

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("극단적 Outlier 분석 (10m 이상)")
    print("=" * 80)

    # 모델 로드
    print(f"\n✅ 모델 로드: {CHECKPOINT}")
    model, ckpt = load_model(CHECKPOINT, device)

    # Test 데이터 로드
    test_file = DATA_DIR / "test.jsonl"
    print(f"✅ 데이터 로드: {test_file}")

    test_samples = []
    with test_file.open() as f:
        for line in f:
            test_samples.append(json.loads(line))

    print(f"   총 {len(test_samples)}개 샘플\n")

    # 분석
    extreme_outliers = []

    print("🔍 분석 중...")
    with torch.no_grad():
        for idx, sample in enumerate(test_samples):
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0).to(device)
            target = sample['target']

            # 예측
            pred = model(features).squeeze(0).cpu().numpy()

            # Denormalize
            pred_x, pred_y = denormalize_coord(pred[0], pred[1])
            true_x, true_y = denormalize_coord(target[0], target[1])

            # 오차 계산
            error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)

            # 극단적 outlier만 수집
            if error >= EXTREME_THRESHOLD:
                error_x = abs(pred_x - true_x)
                error_y = abs(pred_y - true_y)

                # Feature 통계
                feature_stats = analyze_sample_features(sample['features'])

                extreme_outliers.append({
                    'idx': idx,
                    'error': error,
                    'error_x': error_x,
                    'error_y': error_y,
                    'true_x': true_x,
                    'true_y': true_y,
                    'pred_x': pred_x,
                    'pred_y': pred_y,
                    'feature_stats': feature_stats,
                })

    # 결과 출력
    print(f"\n{'='*80}")
    print(f"극단적 Outlier 발견: {len(extreme_outliers)}개 ({len(extreme_outliers)/len(test_samples)*100:.1f}%)")
    print(f"{'='*80}\n")

    if len(extreme_outliers) == 0:
        print("✅ 극단적 outlier 없음!")
        return

    # 오차 순으로 정렬
    extreme_outliers.sort(key=lambda x: x['error'], reverse=True)

    # 통계 분석
    errors = [x['error'] for x in extreme_outliers]
    error_xs = [x['error_x'] for x in extreme_outliers]
    error_ys = [x['error_y'] for x in extreme_outliers]

    print("📊 극단적 Outlier 통계:")
    print(f"   Max 오차:        {max(errors):.2f}m")
    print(f"   Mean 오차:       {np.mean(errors):.2f}m")
    print(f"   Median 오차:     {np.median(errors):.2f}m")
    print(f"   X 방향 평균:     {np.mean(error_xs):.2f}m")
    print(f"   Y 방향 평균:     {np.mean(error_ys):.2f}m")
    print(f"   X/Y 비율:        {np.mean(error_xs)/np.mean(error_ys):.2f}x")
    print()

    # 위치 분포
    true_xs = [x['true_x'] for x in extreme_outliers]
    true_ys = [x['true_y'] for x in extreme_outliers]

    print("📍 위치 분포:")
    print(f"   X 범위: [{min(true_xs):.1f}, {max(true_xs):.1f}]m")
    print(f"   Y 범위: [{min(true_ys):.1f}, {max(true_ys):.1f}]m")
    print(f"   X 평균: {np.mean(true_xs):.1f}m")
    print(f"   Y 평균: {np.mean(true_ys):.1f}m")
    print()

    # Feature 통계
    print("🔬 센서 특성 분석:")
    all_stats = defaultdict(list)
    for outlier in extreme_outliers:
        for key, val in outlier['feature_stats'].items():
            all_stats[key].append(val)

    print(f"   MagX std 평균:      {np.mean(all_stats['magx_std']):.3f}")
    print(f"   MagY std 평균:      {np.mean(all_stats['magy_std']):.3f}")
    print(f"   MagZ std 평균:      {np.mean(all_stats['magz_std']):.3f}")
    print(f"   MagX 변화율 평균:   {np.mean(all_stats['magx_change_rate']):.3f}")
    print(f"   MagY 변화율 평균:   {np.mean(all_stats['magy_change_rate']):.3f}")
    print(f"   MagZ 변화율 평균:   {np.mean(all_stats['magz_change_rate']):.3f}")
    print()

    # 상세 결과 파일로 저장
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)

    with OUTPUT_FILE.open('w') as f:
        f.write("=" * 80 + "\n")
        f.write("극단적 Outlier 상세 분석 (10m 이상)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"총 {len(extreme_outliers)}개 발견 ({len(extreme_outliers)/len(test_samples)*100:.1f}%)\n\n")

        f.write("=" * 80 + "\n")
        f.write("상위 20개 Worst Cases:\n")
        f.write("=" * 80 + "\n\n")

        for i, outlier in enumerate(extreme_outliers[:20], 1):
            f.write(f"[{i}] 샘플 #{outlier['idx']}\n")
            f.write(f"  오차:      {outlier['error']:.2f}m (X: {outlier['error_x']:.2f}m, Y: {outlier['error_y']:.2f}m)\n")
            f.write(f"  실제 위치: ({outlier['true_x']:.2f}, {outlier['true_y']:.2f})\n")
            f.write(f"  예측 위치: ({outlier['pred_x']:.2f}, {outlier['pred_y']:.2f})\n")

            stats = outlier['feature_stats']
            f.write(f"  MagX: mean={stats['magx_mean']:.3f}, std={stats['magx_std']:.3f}, change={stats['magx_change_rate']:.3f}\n")
            f.write(f"  MagY: mean={stats['magy_mean']:.3f}, std={stats['magy_std']:.3f}, change={stats['magy_change_rate']:.3f}\n")
            f.write(f"  MagZ: mean={stats['magz_mean']:.3f}, std={stats['magz_std']:.3f}, change={stats['magz_change_rate']:.3f}\n")
            f.write("\n")

        # 전체 통계
        f.write("=" * 80 + "\n")
        f.write("전체 통계:\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"오차 분포:\n")
        f.write(f"  Max:    {max(errors):.2f}m\n")
        f.write(f"  Mean:   {np.mean(errors):.2f}m\n")
        f.write(f"  Median: {np.median(errors):.2f}m\n")
        f.write(f"  P90:    {np.percentile(errors, 90):.2f}m\n")
        f.write(f"\n")

        f.write(f"방향별 오차:\n")
        f.write(f"  X 평균: {np.mean(error_xs):.2f}m\n")
        f.write(f"  Y 평균: {np.mean(error_ys):.2f}m\n")
        f.write(f"  X/Y 비율: {np.mean(error_xs)/np.mean(error_ys):.2f}x\n")
        f.write(f"\n")

        f.write(f"위치 분포:\n")
        f.write(f"  X 범위: [{min(true_xs):.1f}, {max(true_xs):.1f}]m (평균: {np.mean(true_xs):.1f}m)\n")
        f.write(f"  Y 범위: [{min(true_ys):.1f}, {max(true_ys):.1f}]m (평균: {np.mean(true_ys):.1f}m)\n")

    print(f"✅ 상세 결과 저장: {OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("분석 완료!")
    print("=" * 80)
    print()
    print("💡 다음 단계:")
    print("   1. 위 통계로 패턴 파악")
    print("   2. 특정 위치/구간에 집중되어 있는지 확인")
    print("   3. 센서값이 비정상인지 확인")
    print("   4. 필요시 해당 샘플 제거 또는 Trimmed Loss 적용")

if __name__ == "__main__":
    main()
