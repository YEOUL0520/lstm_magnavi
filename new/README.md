# Clean Hyena Model (MSE Loss 버전)

커밋 0920816 기준의 깔끔한 학습 코드 + 현재 전처리 방식

## 변경사항

### ✅ 유지
- 현재 `nodes_final.csv` (수정된 노드 좌표)
- 현재 전처리 코드의 그리드 선형보간
- 정규화 파라미터: `COORD_CENTER = (-44.3, -0.3)`, `COORD_SCALE = 48.8`

### ❌ 제거
- `WeightedXYLoss` (Huber Loss + X 방향 2배 가중치)
- → 깔끔한 `MSELoss`로 복구

### 이유
RMSE가 손댈 때마다 커져서 순정 MSE Loss로 돌아감

---

## 전체 파이프라인

### 1. 데이터 전처리

```bash
# Raw CSV → Preprocessed CSV (그리드 선형보간 포함)
source venv/bin/activate
python new/src/preprocess_from_csv.py \
  --raw-dir data/raw \
  --nodes-file new/data/nodes_final.csv \
  --output-dir new/data/preprocessed
```

**생성:** `new/data/preprocessed/*.csv`

### 2. Sliding Window 생성

전처리 스크립트에 sliding window 생성이 포함되어 있음.

**Sliding Window 설정:**
- Window size: 250 timesteps
- Stride: 50 (80% overlap, ~0.86m 이동/window)
- 전체 샘플: ~19K개

**Stratified 분할:**
- 각 경로(87개)에서 측정을 비율대로 분할
- 4개 측정: Train 2개, Val 1개, Test 1개
- 5개 측정: Train 3개, Val 1개, Test 1개
- 결과: 모든 87개 경로가 Train/Val/Test에 포함

**생성:** `new/data/sliding_mag4/{train,val,test}.jsonl` (~11K / ~4K / ~4K samples)

### 3. 학습

```bash
# 학습 시작
python new/src/train_sliding.py \
  --data-dir new/data/sliding_mag4 \
  --epochs 100 \
  --batch-size 128 \
  --lr 2e-4 \
  --hidden-dim 384 \
  --depth 10 \
  --patience 15 \
  --checkpoint-dir new/models/hyena_mag4/checkpoints
```

**생성:** `new/models/hyena_mag4/checkpoints/best.pt`

### 4. 테스트

```bash
# 모델 평가
python new/src/test_only.py \
  --checkpoint new/models/hyena_mag4/checkpoints/best.pt \
  --data-dir new/data/sliding_mag4 \
  --hidden-dim 384 \
  --depth 10 \
  --batch-size 128
```

---

## 빠른 시작

```bash
# 1. 전처리 (시간 오래 걸림)
source venv/bin/activate
python new/src/preprocess_from_csv.py \
  --raw-dir data/raw \
  --nodes-file new/data/nodes_final.csv \
  --output-dir new/data/preprocessed

# 2. 학습 (몇 시간)
python new/src/train_sliding.py \
  --data-dir new/data/sliding_mag4 \
  --epochs 100 \
  --batch-size 128 \
  --hidden-dim 384 \
  --depth 10

# 3. 테스트 (몇 분)
python new/src/test_only.py \
  --checkpoint new/models/hyena_mag4/checkpoints/best.pt \
  --data-dir new/data/sliding_mag4 \
  --hidden-dim 384 \
  --depth 10
```

---

## 디렉토리 구조

```
new/
├── README.md                           # 이 파일
├── src/
│   ├── preprocess_from_csv.py          # 전처리 (그리드 선형보간)
│   ├── model.py                        # Hyena 모델
│   ├── train_sliding.py                # 학습 (Clean MSE Loss)
│   └── test_only.py                    # 테스트
├── data/
│   ├── nodes_final.csv                 # 노드 좌표 (수정된 버전)
│   ├── preprocessed/                   # 전처리된 CSV
│   └── sliding_mag4/                   # Sliding Window JSONL
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── meta.json
└── models/
    └── hyena_mag4/
        └── checkpoints/
            └── best.pt                 # 학습된 모델
```

---

## 차이점 비교

| 항목 | 기존 (Huber + 가중치) | New (Clean MSE) |
|------|---------------------|----------------|
| Loss | WeightedXYLoss | nn.MSELoss() |
| X 가중치 | 2.0x | 1.0x |
| Y 가중치 | 1.0x | 1.0x |
| Huber delta | 1.0 | N/A |
| 정규화 | (-44.3, -0.3), 48.8 | 동일 |
| 전처리 | 그리드 선형보간 | 동일 |
| nodes_final.csv | 현재 버전 | 동일 |

---

## 기대 효과

- RMSE 감소 (복잡한 Loss가 오히려 해를 끼쳤음)
- 더 안정적인 학습
- 해석 가능성 향상

---

## 문제 해결

### 오류: "No such file"
```bash
# 디렉토리 생성
mkdir -p new/data/preprocessed
mkdir -p new/data/sliding_mag4
mkdir -p new/models/hyena_mag4/checkpoints
```

### RMSE가 여전히 높음
- 데이터 재확인
- 학습률 조정 (`--lr 1e-4`)
- Depth 조정 (`--depth 8`)

---

## 참고

- 원본 코드: 커밋 0920816
- Huber Loss 제거 이유: RMSE 증가
- 전처리 유지 이유: 그리드 선형보간이 성능 향상에 중요
