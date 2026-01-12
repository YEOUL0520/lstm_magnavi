#!/bin/bash
# 모델 학습

set -e  # 에러 발생시 중단

# 파라미터 받기 (기본값 설정)
EPOCHS=${1:-50}
BATCH_SIZE=${2:-32}
HIDDEN_DIM=${3:-256}
DEPTH=${4:-8}
DROPOUT=${5:-0.1}
PATIENCE=${6:-10}
LR=${7:-2e-4}
WARMUP_EPOCHS=${8:-5}
DEVICE=${9:-mps}

echo "=========================================="
echo "모델 학습 시작"
echo "=========================================="
echo "Epochs:        $EPOCHS"
echo "Batch Size:    $BATCH_SIZE"
echo "Hidden Dim:    $HIDDEN_DIM"
echo "Depth:         $DEPTH"
echo "Dropout:       $DROPOUT"
echo "Patience:      $PATIENCE"
echo "Learning Rate: $LR"
echo "Warmup Epochs: $WARMUP_EPOCHS"
echo "Device:        $DEVICE"
echo ""

# MPS (Apple Silicon) 사용
python3 -c "
import sys
sys.path.append('src')
from train_sliding import train_sliding
from pathlib import Path

train_sliding(
    data_dir=Path('data/sliding_mag4'),
    epochs=$EPOCHS,
    batch_size=$BATCH_SIZE,
    lr=$LR,
    hidden_dim=$HIDDEN_DIM,
    depth=$DEPTH,
    dropout=$DROPOUT,
    patience=$PATIENCE,
    checkpoint_dir=Path('models/hyena_mag4/checkpoints'),
    device='$DEVICE',
    seed=42,
    warmup_epochs=$WARMUP_EPOCHS,
)
"

echo ""
echo "✅ 학습 완료!"
