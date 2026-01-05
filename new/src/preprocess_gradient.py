#!/usr/bin/env python3
"""전처리된 CSV → JSONL 변환 (Sliding Window + Gradient Features)"""
import json
import csv
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pywt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Random seed 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 정규화 기준값
BASE_MAG = (-33.0, -15.0, -42.0)
COORD_CENTER = (-44.3, -0.3)
COORD_SCALE = 48.8

def normalize_mag(val: float, base: float) -> float:
    return (val - base) / 10.0

def normalize_coord(x: float, y: float) -> Tuple[float, float]:
    x_norm = (x - COORD_CENTER[0]) / COORD_SCALE
    y_norm = (y - COORD_CENTER[1]) / COORD_SCALE
    return (x_norm, y_norm)

def wavelet_denoise(signal: List[float], wavelet='db4', level=3) -> List[float]:
    """Wavelet denoising"""
    if len(signal) < 2**level:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet).tolist()

def process_file(args):
    """파일 하나 처리"""
    file_path, window_size, stride = args

    # CSV 읽기
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < window_size:
        return []

    # 신호 추출 및 웨이브렛 디노이징
    magx = [float(row['magx']) for row in rows]
    magy = [float(row['magy']) for row in rows]
    magz = [float(row['magz']) for row in rows]

    magx_denoised = wavelet_denoise(magx)
    magy_denoised = wavelet_denoise(magy)
    magz_denoised = wavelet_denoise(magz)

    # Magnitude 미리 계산
    magnitudes = [
        np.sqrt(magx_denoised[i]**2 + magy_denoised[i]**2 + magz_denoised[i]**2)
        for i in range(len(rows))
    ]

    # Sliding window 생성
    samples = []
    for i in range(0, len(rows) - window_size + 1, stride):
        window_rows = rows[i:i + window_size]

        # Features: 정규화된 센서값 + Gradient
        features = []
        for j, row in enumerate(window_rows):
            idx = i + j

            # 현재 시점 값
            mag_x = magx_denoised[idx]
            mag_y = magy_denoised[idx]
            mag_z = magz_denoised[idx]
            magnitude = magnitudes[idx]

            # Gradient 계산 (첫 timestep은 0)
            if idx == 0 or j == 0:
                delta_x = 0.0
                delta_y = 0.0
                delta_z = 0.0
                delta_mag = 0.0
            else:
                delta_x = mag_x - magx_denoised[idx - 1]
                delta_y = mag_y - magy_denoised[idx - 1]
                delta_z = mag_z - magz_denoised[idx - 1]
                delta_mag = magnitude - magnitudes[idx - 1]

            # 8 features: 기존 4개 + Gradient 4개
            feature_vec = [
                # 기존 features
                normalize_mag(mag_x, BASE_MAG[0]),
                normalize_mag(mag_y, BASE_MAG[1]),
                normalize_mag(mag_z, BASE_MAG[2]),
                normalize_mag(magnitude, 0.0),
                # Gradient features (baseline=0)
                normalize_mag(delta_x, 0.0),
                normalize_mag(delta_y, 0.0),
                normalize_mag(delta_z, 0.0),
                normalize_mag(delta_mag, 0.0),
            ]
            features.append(feature_vec)

        # Target: 윈도우 끝점의 정규화된 좌표
        last_row = window_rows[-1]
        x = float(last_row['x'])
        y = float(last_row['y'])
        x_norm, y_norm = normalize_coord(x, y)

        sample = {
            "features": features,
            "target": [x_norm, y_norm]
        }
        samples.append(sample)

    return samples

def main():
    # 설정
    preprocessed_dir = Path("data/preprocessed")
    output_dir = Path("new/data/sliding_grad")
    output_dir.mkdir(exist_ok=True, parents=True)

    window_size = 250
    stride = 50

    print("=" * 80)
    print("전처리된 CSV → JSONL 변환 (Gradient Features)")
    print("=" * 80)
    print(f"입력 디렉토리: {preprocessed_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"윈도우 크기: {window_size}")
    print(f"스트라이드: {stride}")
    print(f"Features: 8 (MagX/Y/Z, Mag, ΔMagX/Y/Z, ΔMag)")
    print()

    # 캐싱: 기존 전처리 결과 확인
    meta_path = output_dir / "meta.json"
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    if meta_path.exists() and train_path.exists() and val_path.exists() and test_path.exists():
        try:
            with meta_path.open() as f:
                existing_meta = json.load(f)

            # 파라미터 비교
            params_match = (
                existing_meta.get("window_size") == window_size and
                existing_meta.get("stride") == stride and
                existing_meta.get("n_features") == 8
            )

            if params_match:
                print("✅ 전처리가 이미 완료되었습니다!")
                print(f"   출력 디렉토리: {output_dir}")
                print(f"   Train: {existing_meta.get('n_train')}개 샘플")
                print(f"   Val:   {existing_meta.get('n_val')}개 샘플")
                print(f"   Test:  {existing_meta.get('n_test')}개 샘플")
                print()
                print("💡 강제로 재실행하려면 meta.json을 삭제하세요.")
                print("=" * 80)
                return
            else:
                print("⚠️  기존 전처리 결과와 파라미터가 다릅니다. 재실행합니다.")
                print(f"   기존: window_size={existing_meta.get('window_size')}, stride={existing_meta.get('stride')}")
                print(f"   요청: window_size={window_size}, stride={stride}")
                print()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  메타데이터 파일이 손상되었습니다. 재실행합니다. ({e})")
            print()

    # 모든 CSV 파일
    csv_files = sorted(preprocessed_dir.glob("*.csv"))
    print(f"총 {len(csv_files)}개 파일 발견")

    # 경로별로 파일 그룹화
    from collections import defaultdict
    routes = defaultdict(list)
    for f in csv_files:
        parts = f.stem.split("_")
        route_key = f"{parts[0]}_{parts[1]}"
        routes[route_key].append(f)

    print(f"총 {len(routes)}개 경로")
    print()

    # Stratified 분할: 각 경로에서 비율대로 샘플링
    train_files = []
    val_files = []
    test_files = []

    for route_key, files in routes.items():
        random.shuffle(files)
        n = len(files)

        if n == 4:
            # 4개: Train 2개, Val 1개, Test 1개
            train_files.extend(files[:2])
            val_files.append(files[2])
            test_files.append(files[3])
        elif n == 5:
            # 5개: Train 3개, Val 1개, Test 1개
            train_files.extend(files[:3])
            val_files.append(files[3])
            test_files.append(files[4])
        else:
            # 예외 처리 (혹시 다른 개수가 있을 경우)
            n_train = max(1, int(n * 0.6))
            n_val = max(1, int(n * 0.2))
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train + n_val])
            test_files.extend(files[n_train + n_val:])

    print(f"Stratified 분할 완료:")
    print(f"  Train: {len(train_files)}개 파일 ({len(train_files)*100/len(csv_files):.1f}%)")
    print(f"  Val:   {len(val_files)}개 파일 ({len(val_files)*100/len(csv_files):.1f}%)")
    print(f"  Test:  {len(test_files)}개 파일 ({len(test_files)*100/len(csv_files):.1f}%)")
    print(f"  → 모든 {len(routes)}개 경로가 Train/Val/Test에 포함됨")
    print()

    # 멀티프로세싱으로 처리
    n_workers = min(cpu_count(), 8)
    print(f"병렬 처리: {n_workers} workers\n")

    def process_split(files, split_name):
        print(f"처리 중: {split_name}")

        args_list = [(f, window_size, stride) for f in files]

        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_file, args_list),
                total=len(files),
                desc=split_name
            ))

        # 샘플 수집
        all_samples = []
        for samples in results:
            all_samples.extend(samples)

        # JSONL 저장
        output_file = output_dir / f"{split_name}.jsonl"
        with output_file.open('w') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')

        print(f"  {split_name}: {len(all_samples)}개 샘플 저장 → {output_file}")
        return len(all_samples)

    # 각 split 처리
    n_train_samples = process_split(train_files, "train")
    n_val_samples = process_split(val_files, "val")
    n_test_samples = process_split(test_files, "test")

    # 메타데이터 저장
    meta = {
        "n_features": 8,  # magx, magy, magz, magnitude, delta_magx, delta_magy, delta_magz, delta_magnitude
        "window_size": window_size,
        "stride": stride,
        "n_train": n_train_samples,
        "n_val": n_val_samples,
        "n_test": n_test_samples,
    }

    with (output_dir / "meta.json").open('w') as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 80)
    print("✅ 변환 완료!")
    print(f"  출력: {output_dir}")
    print(f"  Features: 8개 (기존 4개 + Gradient 4개)")
    print(f"  Train: {n_train_samples:,}개 샘플")
    print(f"  Val:   {n_val_samples:,}개 샘플")
    print(f"  Test:  {n_test_samples:,}개 샘플")
    print("=" * 80)

if __name__ == "__main__":
    main()
