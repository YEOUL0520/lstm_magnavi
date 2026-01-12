# 모델 디렉토리

학습된 모델 체크포인트를 종류별로 관리합니다.

## 📁 구조

```
models/
├── hyena_mag4/              # Hyena 모델 (mag4 features)
│   └── checkpoints/
│       ├── best.pt          # 최고 성능 모델
│       └── last.pt          # 마지막 epoch 모델
│
└── README.md                # 이 파일
```

## 🎯 현재 모델

### Hyena (mag4)
- **경로**: `models/hyena_mag4/checkpoints/best.pt`
- **Features**: MagX, MagY, MagZ, Magnitude (4개)
- **성능**:
  - MAE: 0.948m
  - P90: 1.660m
  - RMSE: 1.345m

## 🚀 사용 방법

### 학습
```bash
./run_train.sh
# 체크포인트 저장: models/hyena_mag4/checkpoints/
```

### 테스트
```bash
python3 src/test.py --checkpoint models/hyena_mag4/checkpoints/best.pt
```

## 📊 향후 모델 추가 예정

```
models/
├── hyena_mag4/              # 현재 모델
├── hyena_mag3/              # mag3 features (예정)
├── hyena_wavelet_off/       # wavelet 없이 (예정)
└── lstm_baseline/           # LSTM 비교 (예정)
```

---

**Last Updated**: 2025-11-26
**Current Best**: Hyena mag4 (MAE=0.948m, P90=1.660m)
