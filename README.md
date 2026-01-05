# Hyena Indoor Positioning System

지자기 센서 데이터를 이용한 실내 측위 시스템 (Hyena Architecture)

## 🎯 핵심 성과

| 지표 | 목표 | 달성 | 상태 |
|------|------|------|------|
| **P90** | < 2m | **1.660m** | ✅ |
| **MAE** | < 1.4m | **0.948m** | ✅ |
| **Median** | < 1m | **0.552m** | ✅ |
| **RMSE** | < 2.5m | **2.202m** | ✅ |

- 🎯 **90%의 예측이 1.66m 이내 오차**
- 📊 **평균 오차 0.95m**
- 🏆 **중앙값 0.55m**

## 🚀 빠른 시작

### 1. 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행 (한 줄로 끝)

#### 🆕 새 전처리 (권장): 격자 기반 좌표 보간
```bash
./run_new_pipeline.sh
```
- Raw → Preprocessed (격자 0.45m 기반 좌표 추가)
- Preprocessed → JSONL (슬라이딩 윈도우)
- 학습 자동 실행

#### 🔄 구 전처리: 회전 노드 기반 세그먼트 보간
```bash
./run_sliding_window.sh
```
- Raw → JSONL (한 번에 처리)
- 회전 노드 기반 선형 보간
- 학습 자동 실행

### 3. 단계별 실행 (필요시)
```bash
# Step 1: Raw → Preprocessed
python3 scripts/preprocessing/preprocess_all_data.py

# Step 2: Preprocessed → JSONL
bash scripts/run_preprocess.sh

# Step 3: 학습
bash scripts/run_train.sh

# Step 4: 테스트
python3 src/test_only.py \
  --checkpoint models/hyena_mag4/checkpoints/best.pt \
  --data-dir data/sliding_mag4
```

## 📁 프로젝트 구조

```
lstm/
├── run_new_pipeline.sh      # 🆕 새 전처리 전체 파이프라인
├── run_sliding_window.sh    # 🔄 구 전처리 전체 파이프라인
├── README.md                # 메인 문서
├── requirements.txt
│
├── src/                     # 소스 코드
│   ├── model.py                  # Hyena 모델
│   ├── preprocess_from_csv.py    # 새: Preprocessed → JSONL
│   ├── preprocess_sliding.py     # 구: Raw → JSONL
│   ├── train_sliding.py          # 학습
│   └── test_only.py              # 테스트
│
├── scripts/                 # 실행 & 유틸리티 스크립트
│   ├── run_preprocess.sh         # Step 2: CSV → JSONL
│   ├── run_train.sh              # Step 3: 학습
│   ├── run_all.sh                # 전체 (deprecated)
│   └── preprocessing/
│       └── preprocess_all_data.py   # Step 1: Raw → Preprocessed
│
├── models/                  # 모델
│   └── hyena_mag4/
│       └── checkpoints/
│           ├── best.pt
│           └── last.pt
│
├── data/
│   ├── raw/                # 원본 센서 CSV (404개)
│   ├── preprocessed/       # 좌표 추가된 CSV
│   ├── sliding_mag4/       # 학습용 JSONL
│   ├── nodes_final.csv
│   └── node_connections.csv
│
└── analysis/               # 분석 스크립트
    ├── outliers/          # Outlier 분석
    ├── quality/           # 데이터 품질
    ├── distribution/      # 데이터 분포
    ├── performance/       # 모델 성능
    └── basic/             # 기본 분석
```

## 🧠 핵심 기술

- **Hyena Architecture**: O(n log n) 복잡도, FFT 기반 Long Convolution
- **Sliding Window**: 250 steps, stride 25
- **Wavelet Denoising**: db4, level=3
- **Mixed Precision Training (AMP)**
- **Adaptive Learning Rate** (5-epoch moving average)
- **Early Stopping** (P90 기준)

## 📊 데이터셋

- **원본**: 404개 CSV 파일 (87개 경로)
- **전처리 후**: 13,611개 샘플
- **분할**: Train 80%, Val 10%, Test 10%
- **센서**: MagX, MagY, MagZ + Orientation (Yaw, Roll, Pitch)

## 🔧 Git LFS 설정

대용량 파일 관리를 위해 Git LFS 사용:

```bash
# LFS 설치
brew install git-lfs        # macOS
git lfs install

# 저장소 클론
git clone git@github.com:midas-capston-design/hyena.git
cd hyena
```

## 🤝 기여

Midas Capstone Design Team

---

**Last Updated**: 2025-11-26
**Best Model**: MAE=0.948m, P90=1.660m
**Repository**: https://github.com/midas-capston-design/hyena
