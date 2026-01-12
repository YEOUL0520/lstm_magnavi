#!/bin/bash
# Sliding Window 방식: Feature 모드 비교 (mag3 vs mag4 vs full)

set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "🚀 Sliding Window Feature 비교 실험"
echo "========================================="
echo ""

# 공통 설정
WINDOW_SIZE=250
STRIDE=50
EPOCHS=50
BATCH_SIZE=32
HIDDEN_DIM=256
DEPTH=8
PATIENCE=10
LR=2e-4

# ============================================================================
# 1. MAG3 (MagX, MagY, MagZ)
# ============================================================================
echo "========================================="
echo "📊 [1/6] MAG3 전처리..."
echo "========================================="
python3 src/preprocess_sliding.py \
  --raw-dir data/raw \
  --nodes data/nodes_final.csv \
  --output data/sliding_mag3 \
  --feature-mode mag3 \
  --window-size $WINDOW_SIZE \
  --stride $STRIDE

echo ""
echo "========================================="
echo "🧠 [2/6] MAG3 학습..."
echo "========================================="
python3 src/train_sliding.py \
  --data-dir data/sliding_mag3 \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --hidden-dim $HIDDEN_DIM \
  --depth $DEPTH \
  --dropout 0.1 \
  --patience $PATIENCE \
  --checkpoint-dir checkpoints_sliding_mag3

# ============================================================================
# 2. MAG4 (MagX, MagY, MagZ, Magnitude)
# ============================================================================
echo ""
echo "========================================="
echo "📊 [3/6] MAG4 전처리..."
echo "========================================="
python3 src/preprocess_sliding.py \
  --raw-dir data/raw \
  --nodes data/nodes_final.csv \
  --output data/sliding_mag4 \
  --feature-mode mag4 \
  --window-size $WINDOW_SIZE \
  --stride $STRIDE

echo ""
echo "========================================="
echo "🧠 [4/6] MAG4 학습..."
echo "========================================="
python3 src/train_sliding.py \
  --data-dir data/sliding_mag4 \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --hidden-dim $HIDDEN_DIM \
  --depth $DEPTH \
  --dropout 0.1 \
  --patience $PATIENCE \
  --checkpoint-dir checkpoints_sliding_mag4

# ============================================================================
# 3. FULL (MagX, MagY, MagZ, Pitch, Roll, Yaw)
# ============================================================================
echo ""
echo "========================================="
echo "📊 [5/6] FULL 전처리..."
echo "========================================="
python3 src/preprocess_sliding.py \
  --raw-dir data/raw \
  --nodes data/nodes_final.csv \
  --output data/sliding_full \
  --feature-mode full \
  --window-size $WINDOW_SIZE \
  --stride $STRIDE

echo ""
echo "========================================="
echo "🧠 [6/6] FULL 학습..."
echo "========================================="
python3 src/train_sliding.py \
  --data-dir data/sliding_full \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --hidden-dim $HIDDEN_DIM \
  --depth $DEPTH \
  --dropout 0.1 \
  --patience $PATIENCE \
  --checkpoint-dir checkpoints_sliding_full

# ============================================================================
# 결과 요약
# ============================================================================
echo ""
echo "========================================="
echo "📈 실험 완료! 결과 요약"
echo "========================================="
echo ""
echo "1. MAG3 (3 features: MagX, MagY, MagZ)"
echo "   체크포인트: checkpoints_sliding_mag3/best.pt"
echo ""
echo "2. MAG4 (4 features: MagX, MagY, MagZ, Magnitude)"
echo "   체크포인트: checkpoints_sliding_mag4/best.pt"
echo ""
echo "3. FULL (6 features: MagX, MagY, MagZ, Pitch, Roll, Yaw)"
echo "   체크포인트: checkpoints_sliding_full/best.pt"
echo ""
echo "각 모델의 Test RMSE, MAE, Median, P90를 비교하세요!"
echo "========================================="
