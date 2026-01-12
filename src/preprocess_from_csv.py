#!/usr/bin/env python3
"""전처리된 CSV → JSONL 변환 (Sliding Window)

[MODIFIED]
- (논문 반영) 자기장 벡터를 device frame(Mx,My,Mz)에서 world frame(Bxw,Byw,Bzw)로 변환하여
  orientation(roll/pitch/yaw)에 따른 축 값 변화 영향을 줄임.
  - 논문: Magnetic Vector Calibration for Real-Time Indoor Positioning (Son & Choi, 2020)
  - 핵심 아이디어: "자기장 벡터는 3D 벡터이므로 센서 방향이 바뀌면 각 축 값이 달라져 map mismatch가 커짐.
    따라서 회전행렬을 이용해 global coordinate로 변환해 orientation 영향을 보정" :contentReference[oaicite:1]{index=1}

- (주의) 논문 2단계(이동 방향 차이 보정: relative rotation angle θ + circle parametric)는
  '맵 수집 방향'과 'gyro 기반 θ 추정'이 필요해서 Hyena 데이터 구조만으로 1:1 재현이 어려움.
  대신 학습용으로 world frame 벡터 + 안정적인 파생 특징(|B|, Bh, dip)을 추가해
  방향 변화에 더 강건한 입력을 구성함.
"""

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

# =========================
# 기존 정규화 기준값 유지
# =========================
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
    """Wavelet denoising (기존 유지)"""
    if len(signal) < 2**level:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet).tolist()

# =====================================================================
# [ADDED] Euler(Yaw, Pitch, Roll) -> Rotation Matrix (ZYX)
# ---------------------------------------------------------------------
# - yaw, pitch, roll은 degree로 들어온다고 가정하고 rad로 변환해서 사용
# - 회전 순서: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
#   (일반적인 yaw-pitch-roll / ZYX convention)
#
# [논문 반영]
# - 논문은 device local coord를 global coord로 변환하기 위해 rotation matrix R을 사용 :contentReference[oaicite:2]{index=2}
# - 여기서는 주어진 yaw/pitch/roll을 이용해 R을 구성해 B_world = R @ B_device 수행
# =====================================================================
def euler_ypr_to_R(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)

    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float32)

    Rx = np.array([[1.0, 0.0,  0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)

# =====================================================================
# [ADDED] World-frame features builder
# ---------------------------------------------------------------------
# - input: denoised mag (mx,my,mz) + yaw/roll/pitch(deg)
# - output: world vector (Bxw,Byw,Bzw) + derived scalars(|B|, Bh, dip)
#
# [논문 반영 포인트]
# - magnitude만 쓰면 uniqueness가 줄어 정확도가 떨어질 수 있다고 지적 :contentReference[oaicite:3]{index=3}
# - 그래서 벡터를 쓰되, orientation 영향을 줄이기 위해 global로 변환 후
#   vector + scalar를 함께 제공(학습 안정성과 방향 강건성 개선 목적)
# =====================================================================
def build_world_features(mx: float, my: float, mz: float,
                         yaw_deg: float, pitch_deg: float, roll_deg: float) -> Tuple[float, float, float, float, float, float]:
    # degree -> rad
    yaw = yaw_deg * np.pi / 180.0
    pitch = pitch_deg * np.pi / 180.0
    roll = roll_deg * np.pi / 180.0

    R = euler_ypr_to_R(yaw, pitch, roll)

    B_device = np.array([mx, my, mz], dtype=np.float32)
    B_world = R @ B_device  # [Bxw, Byw, Bzw]

    bxw, byw, bzw = float(B_world[0]), float(B_world[1]), float(B_world[2])

    # derived scalars
    B_mag = float(np.sqrt(bxw * bxw + byw * byw + bzw * bzw) + 1e-8)
    Bh = float(np.sqrt(bxw * bxw + byw * byw) + 1e-8)
    dip = float(np.arctan2(bzw, Bh))  # radians

    return bxw, byw, bzw, B_mag, Bh, dip

def process_file(args):
    """파일 하나 처리"""
    file_path, window_size, stride = args

    # CSV 읽기
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < window_size:
        return []

    # 신호 추출 및 웨이브렛 디노이징 (기존 유지)
    magx = [float(row['magx']) for row in rows]
    magy = [float(row['magy']) for row in rows]
    magz = [float(row['magz']) for row in rows]

    magx_denoised = wavelet_denoise(magx)
    magy_denoised = wavelet_denoise(magy)
    magz_denoised = wavelet_denoise(magz)

    # Sliding window 생성
    samples = []
    for i in range(0, len(rows) - window_size + 1, stride):
        window_rows = rows[i:i + window_size]

        # ============================================================
        # [MODIFIED] Features 구성 변경
        # ------------------------------------------------------------
        # 기존: (magx,magy,magz) + (yaw,roll,pitch)를 그대로 feature로 사용
        # 변경: (Bxw,Byw,Bzw) + (|B|, Bh, dip) 를 feature로 사용
        #      - 논문 1단계 반영: orientation에 따른 축 값 변화 감소 :contentReference[oaicite:4]{index=4}
        #      - magnitude만 단독 사용의 한계(uniqueness 저하) 회피 :contentReference[oaicite:5]{index=5}
        #
        # NOTE:
        # - yaw/pitch/roll은 "회전 보정 계산"에만 사용하고 feature에는 넣지 않음
        #   (원하면 ablation으로 feature에 다시 추가하는 버전도 만들 수 있음)
        # - dip은 radian 값이므로 스케일 맞추기 위해 /pi로 [-1,1] 근사 정규화
        # ============================================================
        features = []
        for j, row in enumerate(window_rows):
            idx = i + j

            mx = float(magx_denoised[idx])
            my = float(magy_denoised[idx])
            mz = float(magz_denoised[idx])

            yaw_deg = float(row['yaw'])
            roll_deg = float(row['roll'])
            pitch_deg = float(row['pitch'])

            # world-frame vector + scalars
            bxw, byw, bzw, B_mag, Bh, dip = build_world_features(
                mx, my, mz,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg
            )

            # 정규화:
            # - world vector는 기존 normalize_mag 기준을 그대로 적용(베이스라인과 비교 용이)
            # - |B|, Bh는 대략 50uT 스케일을 고려해 /50으로 스케일링 (너희 데이터에 맞게 조정 가능)
            # - dip은 [-pi/2, pi/2] 정도 범위라 /pi로 [-0.5,0.5] 근사
            feature_vec = [
                normalize_mag(bxw, BASE_MAG[0]),
                normalize_mag(byw, BASE_MAG[1]),
                normalize_mag(bzw, BASE_MAG[2]),
                B_mag / 50.0,
                Bh / 50.0,
                dip / np.pi,
            ]
            features.append(feature_vec)

        # Target: 윈도우 끝점의 정규화된 좌표 (기존 유지)
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
    output_dir = Path("data/sliding_lstm")  # LSTM을 위한 벡터 보정 버전을 저장할 폴더를 따로 생성함
    output_dir.mkdir(exist_ok=True, parents=True)

    window_size = 250
    stride = 25

    print("=" * 80)
    print("전처리된 CSV → JSONL 변환 (Vector-world features)")
    print("=" * 80)
    print(f"입력 디렉토리: {preprocessed_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"윈도우 크기: {window_size}")
    print(f"스트라이드: {stride}")
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

            params_match = (
                existing_meta.get("window_size") == window_size and
                existing_meta.get("stride") == stride and
                existing_meta.get("n_features") == 6
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

    # Train/Val/Test 분할 (6:2:2)
    random.shuffle(csv_files)
    n_train = int(len(csv_files) * 0.6)
    n_val = int(len(csv_files) * 0.2)

    train_files = csv_files[:n_train]
    val_files = csv_files[n_train:n_train + n_val]
    test_files = csv_files[n_train + n_val:]

    print(f"Train: {len(train_files)}개")
    print(f"Val:   {len(val_files)}개")
    print(f"Test:  {len(test_files)}개")
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
        "n_features": 6,  # bxw, byw, bzw, |B|, Bh, dip
        "window_size": window_size,
        "stride": stride,
        "n_train": n_train_samples,
        "n_val": n_val_samples,
        "n_test": n_test_samples,
        "feature_desc": [
            "Bx_world_norm",
            "By_world_norm",
            "Bz_world_norm",
            "|B|/50",
            "Bh/50",
            "dip/pi"
        ],
        "note": "Vector-world features: device mag rotated to global coord using yaw/pitch/roll (paper-inspired).",
    }

    with (output_dir / "meta.json").open('w') as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 80)
    print("✅ 변환 완료!")
    print(f"  출력: {output_dir}")
    print(f"  Train: {n_train_samples:,}개 샘플")
    print(f"  Val:   {n_val_samples:,}개 샘플")
    print(f"  Test:  {n_test_samples:,}개 샘플")
    print("=" * 80)

if __name__ == "__main__":
    main()