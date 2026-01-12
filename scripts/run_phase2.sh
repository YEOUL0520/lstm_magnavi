#!/usr/bin/env bash
# Phase 2: Gradient Features 추가
# 8 features: MagX, MagY, MagZ, Magnitude + Gradients

set -e

PROJECT_ROOT="/Users/yunho/school/lstm"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Phase 2: Gradient Features"
echo "======================================================================"
echo ""
echo "📋 설정:"
echo "  - Features: 8 (기존 4개 + Gradient 4개)"
echo "    * MagX, MagY, MagZ, Magnitude"
echo "    * ΔMagX, ΔMagY, ΔMagZ, ΔMagnitude"
echo "  - LR: 1e-4"
echo "  - Epochs: 400"
echo "  - Patience: 20"
echo ""
echo "======================================================================"

# 1. 가상환경 활성화
echo ""
echo "✅ Step 1: 가상환경 활성화"
source venv/bin/activate

# 2. 전처리 (Gradient features 포함)
echo ""
echo "✅ Step 2: 전처리 (Gradient Features)"
if [ -f "new/data/sliding_grad/meta.json" ]; then
    echo "   ℹ️  전처리 데이터 이미 존재 → Skip"
    echo "   💡 강제 재실행: rm new/data/sliding_grad/meta.json"
else
    echo "   🔄 전처리 시작 (Gradient features 추가)..."
    python new/src/preprocess_gradient.py
    echo "   ✅ 전처리 완료"
fi

# 3. 출력 디렉토리 생성
echo ""
echo "✅ Step 3: 출력 디렉토리 생성"
mkdir -p new/models/phase2/checkpoints
mkdir -p new/models/phase2/logs

# 4. 학습
echo ""
echo "======================================================================"
echo "🔥 Step 4: 학습 시작 (예상 시간: 8-10시간)"
echo "======================================================================"
echo ""

python new/src/train_sliding.py \
    --data-dir new/data/sliding_grad \
    --epochs 400 \
    --batch-size 128 \
    --lr 1e-4 \
    --warmup-epochs 10 \
    --patience 20 \
    --hidden-dim 384 \
    --depth 12 \
    --checkpoint-dir new/models/phase2/checkpoints \
    2>&1 | tee new/models/phase2/logs/train.log

echo ""
echo "======================================================================"
echo "✅ 학습 완료!"
echo "======================================================================"

# 5. 테스트
echo ""
echo "✅ Step 5: 테스트"
python new/src/test_only.py \
    --checkpoint new/models/phase2/checkpoints/best.pt \
    --data-dir new/data/sliding_grad \
    --hidden-dim 384 \
    --depth 12 \
    --batch-size 128 \
    2>&1 | tee new/models/phase2/logs/test.log

echo ""
echo "======================================================================"
echo "🎉 Phase 2 완료!"
echo "======================================================================"
echo ""
echo "📊 Phase 1 vs Phase 2 비교:"
echo "  Phase 1 결과: new/models/phase1/logs/test.log"
echo "  Phase 2 결과: new/models/phase2/logs/test.log"
echo ""
echo "📁 다음 단계:"
echo "  ./scripts/run_phase3.sh  # Outlier Removal"
echo ""
