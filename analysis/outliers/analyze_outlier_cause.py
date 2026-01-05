#!/usr/bin/env python3
"""Outlier 원인 분석 스크립트"""
import json
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# 한글 폰트 설정
def setup_korean_font():
    import platform
    system = platform.system()
    font_candidates = []

    if system == 'Darwin':
        font_candidates = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic']
    elif system == 'Windows':
        font_candidates = ['Malgun Gothic', 'NanumGothic', 'Gulim']
    else:
        font_candidates = ['NanumGothic', 'NanumBarunGothic', 'UnDotum', 'DejaVu Sans']

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 한글 폰트 설정: {font}")
            return True

    print("⚠️  한글 폰트를 찾을 수 없습니다. 기본 폰트로 표시됩니다.")
    plt.rcParams['axes.unicode_minus'] = False
    return False

setup_korean_font()

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))
from model import HyenaPositioning

# 역정규화
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0

def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (x, y)

def analyze_outlier_causes(
    checkpoint_path: Path,
    data_dir: Path,
    threshold: float = 3.0,
    output_dir: Path = Path("analysis/outputs"),
):
    """Outlier 원인 상세 분석"""

    print("=" * 80)
    print("🔍 Outlier 원인 분석")
    print("=" * 80)
    print(f"  Threshold: {threshold}m")
    print()

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Device: {device}")

    # 체크포인트 로드
    print(f"📂 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = checkpoint["meta"]

    model = HyenaPositioning(
        input_dim=meta["n_features"],
        hidden_dim=384,
        output_dim=2,
        depth=10,
        dropout=0.1,
        num_edge_types=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Model loaded")
    print()

    # 테스트 데이터 로드
    test_path = data_dir / "test.jsonl"
    samples = []
    with test_path.open() as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"📊 Total samples: {len(samples)}")
    print()

    # 예측 및 분석 데이터 수집
    print("🔄 Analyzing...")
    results = []

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(samples, desc="Processing", ncols=80)):
            features_array = np.array(sample["features"])  # [250, n_features]
            features = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor(sample["target"], dtype=torch.float32).to(device)

            # 예측
            edge_ids = torch.zeros(1, dtype=torch.long, device=device)
            outputs = model(features, edge_ids)
            pred = outputs[0, -1, :].cpu().numpy()
            target_np = target.cpu().numpy()

            # 역정규화
            pred_pos = denormalize_coord(pred[0], pred[1])
            target_pos = denormalize_coord(target_np[0], target_np[1])

            # Manhattan distance
            dist = abs(pred_pos[0] - target_pos[0]) + abs(pred_pos[1] - target_pos[1])

            # Features 통계
            mag_x = features_array[:, 0]
            mag_y = features_array[:, 1]
            mag_z = features_array[:, 2]

            # 자기장 변화량 (표준편차)
            mag_x_std = np.std(mag_x)
            mag_y_std = np.std(mag_y)
            mag_z_std = np.std(mag_z)
            mag_total_std = np.sqrt(mag_x_std**2 + mag_y_std**2 + mag_z_std**2)

            # 자기장 평균
            mag_x_mean = np.mean(mag_x)
            mag_y_mean = np.mean(mag_y)
            mag_z_mean = np.mean(mag_z)

            # 자기장 변화율 (연속된 값의 차이)
            mag_x_diff = np.diff(mag_x)
            mag_y_diff = np.diff(mag_y)
            mag_z_diff = np.diff(mag_z)
            mag_change_rate = np.mean(np.abs(mag_x_diff)) + np.mean(np.abs(mag_y_diff)) + np.mean(np.abs(mag_z_diff))

            results.append({
                "sample_idx": idx,
                "distance": dist,
                "pred_x": pred_pos[0],
                "pred_y": pred_pos[1],
                "target_x": target_pos[0],
                "target_y": target_pos[1],
                "error_x": abs(pred_pos[0] - target_pos[0]),
                "error_y": abs(pred_pos[1] - target_pos[1]),
                # 센서 특성
                "mag_x_mean": mag_x_mean,
                "mag_y_mean": mag_y_mean,
                "mag_z_mean": mag_z_mean,
                "mag_x_std": mag_x_std,
                "mag_y_std": mag_y_std,
                "mag_z_std": mag_z_std,
                "mag_total_std": mag_total_std,
                "mag_change_rate": mag_change_rate,
            })

    print()

    # Outlier 필터링
    outliers = [r for r in results if r["distance"] > threshold]
    normal = [r for r in results if r["distance"] <= threshold]

    print(f"  Normal: {len(normal)} ({len(normal)/len(results)*100:.1f}%)")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(results)*100:.1f}%)")
    print()

    if len(outliers) == 0:
        print("✅ No outliers found!")
        return

    # === 원인 분석 ===

    # 전체 데이터의 X, Y 범위 계산 (상대 오차 계산용)
    all_targets_x = [r["target_x"] for r in results]
    all_targets_y = [r["target_y"] for r in results]
    x_range = max(all_targets_x) - min(all_targets_x)
    y_range = max(all_targets_y) - min(all_targets_y)

    print("📏 전체 데이터 범위:")
    print(f"  X 범위: {min(all_targets_x):.2f}m ~ {max(all_targets_x):.2f}m (총 {x_range:.2f}m)")
    print(f"  Y 범위: {min(all_targets_y):.2f}m ~ {max(all_targets_y):.2f}m (총 {y_range:.2f}m)")
    print()

    # 1. 위치별 분석
    print("=" * 80)
    print("📍 1. 위치별 분석")
    print("=" * 80)

    outlier_targets_x = [o["target_x"] for o in outliers]
    outlier_targets_y = [o["target_y"] for o in outliers]

    print(f"  Outlier 위치 범위:")
    print(f"    X: {min(outlier_targets_x):.2f}m ~ {max(outlier_targets_x):.2f}m")
    print(f"    Y: {min(outlier_targets_y):.2f}m ~ {max(outlier_targets_y):.2f}m")
    print()

    # X 위치별 분포
    x_bins = defaultdict(int)
    for x in outlier_targets_x:
        bin_label = f"{int(x/5)*5}~{int(x/5)*5+5}m"
        x_bins[bin_label] += 1

    print(f"  X 위치 분포 (5m 구간):")
    for bin_label in sorted(x_bins.keys()):
        count = x_bins[bin_label]
        print(f"    {bin_label:<15}: {count:3d}개 ({count/len(outliers)*100:.1f}%)")
    print()

    # 2. X/Y 오차 방향 분석
    print("=" * 80)
    print("📊 2. 오차 방향 분석 (절대 오차 vs 상대 오차)")
    print("=" * 80)

    outlier_x_errors = [o["error_x"] for o in outliers]
    outlier_y_errors = [o["error_y"] for o in outliers]

    # 상대 오차 계산 (범위 대비 %)
    x_error_mean = np.mean(outlier_x_errors)
    y_error_mean = np.mean(outlier_y_errors)
    x_error_relative = (x_error_mean / x_range) * 100
    y_error_relative = (y_error_mean / y_range) * 100

    x_dominant = sum(1 for o in outliers if o["error_x"] > o["error_y"])
    y_dominant = len(outliers) - x_dominant

    print(f"  X 방향 오차:")
    print(f"    절대 평균: {x_error_mean:.3f}m")
    print(f"    상대 평균: {x_error_relative:.1f}% (전체 X 범위 {x_range:.1f}m 대비)")
    print(f"    최대: {max(outlier_x_errors):.3f}m")
    print(f"    우세 샘플: {x_dominant}개 ({x_dominant/len(outliers)*100:.1f}%)")
    print()
    print(f"  Y 방향 오차:")
    print(f"    절대 평균: {y_error_mean:.3f}m")
    print(f"    상대 평균: {y_error_relative:.1f}% (전체 Y 범위 {y_range:.1f}m 대비)")
    print(f"    최대: {max(outlier_y_errors):.3f}m")
    print(f"    우세 샘플: {y_dominant}개 ({y_dominant/len(outliers)*100:.1f}%)")
    print()

    # 상대 오차 비교
    if x_error_relative > y_error_relative * 1.5:
        print(f"  ⚠️  X 방향 상대 오차({x_error_relative:.1f}%)가 Y 방향({y_error_relative:.1f}%)보다 {x_error_relative/y_error_relative:.1f}배 높음")
        print(f"      → X 방향 예측 성능이 실제로 더 나쁨")
    elif y_error_relative > x_error_relative * 1.5:
        print(f"  ⚠️  Y 방향 상대 오차({y_error_relative:.1f}%)가 X 방향({x_error_relative:.1f}%)보다 {y_error_relative/x_error_relative:.1f}배 높음")
        print(f"      → Y 방향 예측 성능이 실제로 더 나쁨")
    else:
        print(f"  ✅ X, Y 방향 상대 오차가 비슷함 ({x_error_relative:.1f}% vs {y_error_relative:.1f}%)")
        print(f"      → 양 방향 모두 비슷한 수준의 어려움")
    print()

    # 3. 센서 데이터 특성 비교
    print("=" * 80)
    print("🔬 3. 센서 데이터 특성 비교 (Outlier vs Normal)")
    print("=" * 80)

    # Outlier 통계
    outlier_mag_x_std = [o["mag_x_std"] for o in outliers]
    outlier_mag_y_std = [o["mag_y_std"] for o in outliers]
    outlier_mag_z_std = [o["mag_z_std"] for o in outliers]
    outlier_mag_total_std = [o["mag_total_std"] for o in outliers]
    outlier_mag_change = [o["mag_change_rate"] for o in outliers]

    # Normal 통계
    normal_mag_x_std = [n["mag_x_std"] for n in normal]
    normal_mag_y_std = [n["mag_y_std"] for n in normal]
    normal_mag_z_std = [n["mag_z_std"] for n in normal]
    normal_mag_total_std = [n["mag_total_std"] for n in normal]
    normal_mag_change = [n["mag_change_rate"] for n in normal]

    print(f"  자기장 표준편차 (변동성):")
    print(f"    MagX Std:")
    print(f"      Outlier: {np.mean(outlier_mag_x_std):.4f} ± {np.std(outlier_mag_x_std):.4f}")
    print(f"      Normal:  {np.mean(normal_mag_x_std):.4f} ± {np.std(normal_mag_x_std):.4f}")
    print(f"      차이:    {(np.mean(outlier_mag_x_std) - np.mean(normal_mag_x_std))/np.mean(normal_mag_x_std)*100:+.1f}%")
    print()
    print(f"    MagY Std:")
    print(f"      Outlier: {np.mean(outlier_mag_y_std):.4f} ± {np.std(outlier_mag_y_std):.4f}")
    print(f"      Normal:  {np.mean(normal_mag_y_std):.4f} ± {np.std(normal_mag_y_std):.4f}")
    print(f"      차이:    {(np.mean(outlier_mag_y_std) - np.mean(normal_mag_y_std))/np.mean(normal_mag_y_std)*100:+.1f}%")
    print()
    print(f"    MagZ Std:")
    print(f"      Outlier: {np.mean(outlier_mag_z_std):.4f} ± {np.std(outlier_mag_z_std):.4f}")
    print(f"      Normal:  {np.mean(normal_mag_z_std):.4f} ± {np.std(normal_mag_z_std):.4f}")
    print(f"      차이:    {(np.mean(outlier_mag_z_std) - np.mean(normal_mag_z_std))/np.mean(normal_mag_z_std)*100:+.1f}%")
    print()
    print(f"    Total Std:")
    print(f"      Outlier: {np.mean(outlier_mag_total_std):.4f} ± {np.std(outlier_mag_total_std):.4f}")
    print(f"      Normal:  {np.mean(normal_mag_total_std):.4f} ± {np.std(normal_mag_total_std):.4f}")
    print(f"      차이:    {(np.mean(outlier_mag_total_std) - np.mean(normal_mag_total_std))/np.mean(normal_mag_total_std)*100:+.1f}%")
    print()
    print(f"  자기장 변화율 (급격한 변화):")
    print(f"    Outlier: {np.mean(outlier_mag_change):.4f} ± {np.std(outlier_mag_change):.4f}")
    print(f"    Normal:  {np.mean(normal_mag_change):.4f} ± {np.std(normal_mag_change):.4f}")
    print(f"    차이:    {(np.mean(outlier_mag_change) - np.mean(normal_mag_change))/np.mean(normal_mag_change)*100:+.1f}%")
    print()

    # 4. 시각화
    print("📊 시각화 생성 중...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    # (1,1) 위치별 히트맵
    ax1 = plt.subplot(3, 3, 1)
    h = ax1.hist2d(outlier_targets_x, outlier_targets_y, bins=20, cmap='Reds')
    plt.colorbar(h[3], ax=ax1, label='개수')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Outlier 위치 히트맵')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # (1,2) X 오차 히스토그램
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(outlier_x_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(np.mean(outlier_x_errors), color='black', linestyle='--',
                label=f'평균: {np.mean(outlier_x_errors):.2f}m')
    ax2.set_xlabel('X 오차 (m)')
    ax2.set_ylabel('개수')
    ax2.set_title('X 방향 오차 분포')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (1,3) Y 오차 히스토그램
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(outlier_y_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(np.mean(outlier_y_errors), color='black', linestyle='--',
                label=f'평균: {np.mean(outlier_y_errors):.2f}m')
    ax3.set_xlabel('Y 오차 (m)')
    ax3.set_ylabel('개수')
    ax3.set_title('Y 방향 오차 분포')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (2,1) MagX Std 비교
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist([normal_mag_x_std, outlier_mag_x_std], bins=30,
             label=['Normal', 'Outlier'], color=['green', 'red'], alpha=0.6)
    ax4.set_xlabel('MagX Std')
    ax4.set_ylabel('개수')
    ax4.set_title('MagX 표준편차 비교')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # (2,2) MagY Std 비교
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist([normal_mag_y_std, outlier_mag_y_std], bins=30,
             label=['Normal', 'Outlier'], color=['green', 'red'], alpha=0.6)
    ax5.set_xlabel('MagY Std')
    ax5.set_ylabel('개수')
    ax5.set_title('MagY 표준편차 비교')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # (2,3) MagZ Std 비교
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist([normal_mag_z_std, outlier_mag_z_std], bins=30,
             label=['Normal', 'Outlier'], color=['green', 'red'], alpha=0.6)
    ax6.set_xlabel('MagZ Std')
    ax6.set_ylabel('개수')
    ax6.set_title('MagZ 표준편차 비교')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # (3,1) Total Std 비교
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist([normal_mag_total_std, outlier_mag_total_std], bins=30,
             label=['Normal', 'Outlier'], color=['green', 'red'], alpha=0.6)
    ax7.set_xlabel('Total Mag Std')
    ax7.set_ylabel('개수')
    ax7.set_title('전체 자기장 표준편차 비교')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # (3,2) 변화율 비교
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist([normal_mag_change, outlier_mag_change], bins=30,
             label=['Normal', 'Outlier'], color=['green', 'red'], alpha=0.6)
    ax8.set_xlabel('자기장 변화율')
    ax8.set_ylabel('개수')
    ax8.set_title('자기장 변화율 비교')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # (3,3) 오차 vs 자기장 변동성 산점도
    ax9 = plt.subplot(3, 3, 9)
    distances = [o["distance"] for o in outliers]
    ax9.scatter(outlier_mag_total_std, distances, alpha=0.6, c=distances, cmap='Reds')
    ax9.set_xlabel('자기장 변동성 (Total Std)')
    ax9.set_ylabel('오차 (m)')
    ax9.set_title('오차 vs 자기장 변동성')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "outlier_cause_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    print()

    # === 결론 ===
    print("=" * 80)
    print("📝 분석 결론")
    print("=" * 80)

    # 센서 차이 계산
    mag_std_diff = (np.mean(outlier_mag_total_std) - np.mean(normal_mag_total_std)) / np.mean(normal_mag_total_std) * 100
    mag_change_diff = (np.mean(outlier_mag_change) - np.mean(normal_mag_change)) / np.mean(normal_mag_change) * 100

    print(f"1. 위치 특성:")
    print(f"   - Outlier는 X={min(outlier_targets_x):.1f}~{max(outlier_targets_x):.1f}m 범위에 분포")
    print(f"   - 특정 위치에 집중되어 있는지 히트맵 확인 필요")
    print()
    print(f"2. 오차 방향 (상대 오차 기준):")
    print(f"   - X 방향: 절대 {x_error_mean:.2f}m, 상대 {x_error_relative:.1f}% (범위 {x_range:.1f}m 대비)")
    print(f"   - Y 방향: 절대 {y_error_mean:.2f}m, 상대 {y_error_relative:.1f}% (범위 {y_range:.1f}m 대비)")

    if x_error_relative > y_error_relative * 1.5:
        print(f"   ⚠️  X 방향이 실제로 {x_error_relative/y_error_relative:.1f}배 더 나쁨 → X 방향 개선 필요")
    elif y_error_relative > x_error_relative * 1.5:
        print(f"   ⚠️  Y 방향이 실제로 {y_error_relative/x_error_relative:.1f}배 더 나쁨 → Y 방향 개선 필요")
    else:
        print(f"   ✅ 양 방향 성능 비슷함 → 전반적 개선 필요")
    print()
    print(f"3. 센서 데이터 특성:")
    print(f"   - 자기장 변동성: Outlier가 Normal보다 {mag_std_diff:+.1f}% {'높음' if mag_std_diff > 0 else '낮음'}")
    print(f"   - 자기장 변화율: Outlier가 Normal보다 {mag_change_diff:+.1f}% {'높음' if mag_change_diff > 0 else '낮음'}")

    if abs(mag_std_diff) > 10 or abs(mag_change_diff) > 10:
        print(f"   ⚠️  센서 데이터 특성이 10% 이상 차이 → 센서 노이즈/불안정성 의심")
    else:
        print(f"   ✅ 센서 데이터 특성은 유사 → 구조적 문제 가능성 (특정 위치/경로)")

    print()
    print("=" * 80)
    print("✅ 원인 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/hyena_mag4/checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/sliding_mag4")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--output-dir", type=str, default="analysis/outputs")

    args = parser.parse_args()

    analyze_outlier_causes(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        threshold=args.threshold,
        output_dir=Path(args.output_dir),
    )
