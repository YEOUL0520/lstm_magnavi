#!/bin/bash
# 새 전처리 방식 전체 파이프라인: Raw → Preprocessed → JSONL → 학습

set -e  # 에러 발생시 중단

echo "=========================================="
echo "🆕 새 전처리 방식 전체 파이프라인"
echo "=========================================="
echo ""

# ========== 설정 ==========
FEATURE_MODE="mag4"      # mag3, mag4, full
WINDOW_SIZE=250
STRIDE=25                # 새 전처리는 stride 20 사용
EPOCHS=400                # 충분한 학습 시간 확보
BATCH_SIZE=64            # MPS 활용
HIDDEN_DIM=384           # 모델 용량
DEPTH=10                  # 깊이
DROPOUT=0.1              # 드롭아웃
PATIENCE=15              # Early stopping patience
LR=2e-4                  # Learning rate
WARMUP_EPOCHS=10          # Warmup epochs

# Device 자동 감지 (cuda > mps > cpu)
DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")

echo "========== 설정 확인 =========="
echo "Feature Mode:   $FEATURE_MODE"
echo "Window Size:    $WINDOW_SIZE"
echo "Stride:         $STRIDE"
echo "Epochs:         $EPOCHS"
echo "Batch Size:     $BATCH_SIZE"
echo "Hidden Dim:     $HIDDEN_DIM"
echo "Depth:          $DEPTH"
echo "Dropout:        $DROPOUT"
echo "Patience:       $PATIENCE"
echo "Learning Rate:  $LR"
echo "Warmup Epochs:  $WARMUP_EPOCHS"
echo "Device:         $DEVICE"
echo "=============================="
echo ""

# Step 1: Raw → Preprocessed (격자 기반 좌표 추가)
echo "📍 Step 1/3: Raw → Preprocessed (격자 기반 좌표 추가)"
python3 scripts/preprocessing/preprocess_all_data.py

echo ""
echo ""

# Step 2: Preprocessed → JSONL (슬라이딩 윈도우)
echo "📊 Step 2/3: Preprocessed → JSONL (슬라이딩 윈도우)"
bash scripts/run_preprocess.sh $WINDOW_SIZE $STRIDE

echo ""
echo ""

# Step 3: 학습
echo "🚀 Step 3/3: 모델 학습"
bash scripts/run_train.sh \
  $EPOCHS $BATCH_SIZE $HIDDEN_DIM $DEPTH $DROPOUT $PATIENCE $LR $WARMUP_EPOCHS $DEVICE

echo ""
echo "=========================================="
echo "✅ 새 전처리 방식 전체 파이프라인 완료!"
echo "=========================================="
echo ""
echo "체크포인트: models/hyena_mag4/checkpoints/best.pt"
echo "=========================================="
