#!/usr/bin/env bash
# Phase 4: Attention + Multi-scale
# (구현 필요: Multi-scale 전처리 + 모델)

set -e

PROJECT_ROOT="/Users/yunho/school/lstm"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Phase 4: Attention + Multi-scale"
echo "======================================================================"
echo ""
echo "📋 설정:"
echo "  - Features: 8 (Gradient)"
echo "  - Architecture: Multi-scale (100 + 250) + Attention"
echo "  - LR: 1e-4"
echo "  - Epochs: 400"
echo "  - Patience: 25"
echo ""
echo "⚠️  주의: 이 Phase는 새로운 구현이 필요합니다."
echo "   - new/src/preprocess_multiscale.py (구현 필요)"
echo "   - new/src/train_multiscale.py (구현 필요)"
echo ""
echo "======================================================================"

# 1. 가상환경 활성화
echo ""
echo "✅ Step 1: 가상환경 활성화"
source venv/bin/activate

# 2. 구현 파일 체크
echo ""
echo "✅ Step 2: 구현 파일 확인"
MISSING=0

if [ ! -f "new/src/preprocess_multiscale.py" ]; then
    echo "   ❌ new/src/preprocess_multiscale.py 없음"
    MISSING=1
fi

if [ ! -f "new/src/train_multiscale.py" ]; then
    echo "   ❌ new/src/train_multiscale.py 없음"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "======================================================================"
    echo "⚠️  Phase 4는 아직 구현되지 않았습니다."
    echo "======================================================================"
    echo ""
    echo "📝 필요한 작업:"
    echo "  1. Multi-scale 전처리 구현 (100 + 250 timesteps)"
    echo "  2. Multi-scale 모델 구현 (Attention 포함)"
    echo ""
    echo "💡 Phase 3까지 완료 후 구현하세요."
    echo "   Phase 3 결과가 만족스러우면 Phase 4 진행"
    echo ""
    exit 1
fi

# 3. 전처리 (Multi-scale)
echo ""
echo "✅ Step 3: 전처리 (Multi-scale)"
if [ -f "new/data/sliding_multiscale/meta.json" ]; then
    echo "   ℹ️  전처리 데이터 이미 존재 → Skip"
else
    echo "   🔄 전처리 시작 (Multi-scale: 100 + 250)..."
    python new/src/preprocess_multiscale.py \
        --raw-dir data/raw \
        --nodes-file new/data/nodes_final.csv \
        --output-dir new/data/preprocessed_multiscale \
        --window-sizes 100 250
    echo "   ✅ 전처리 완료"
fi

# 4. 출력 디렉토리 생성
echo ""
echo "✅ Step 4: 출력 디렉토리 생성"
mkdir -p new/models/phase4/checkpoints
mkdir -p new/models/phase4/logs

# 5. 학습 (Multi-scale 모델)
echo ""
echo "======================================================================"
echo "🔥 Step 5: 학습 시작 (Multi-scale + Attention)"
echo "======================================================================"
echo ""

python new/src/train_multiscale.py \
    --data-dir new/data/sliding_multiscale \
    --epochs 400 \
    --batch-size 128 \
    --lr 1e-4 \
    --warmup-epochs 10 \
    --patience 25 \
    --hidden-dim 384 \
    --depth 10 \
    --checkpoint-dir new/models/phase4/checkpoints \
    2>&1 | tee new/models/phase4/logs/train.log

echo ""
echo "======================================================================"
echo "✅ 학습 완료!"
echo "======================================================================"

# 6. 테스트
echo ""
echo "✅ Step 6: 테스트"
python new/src/test_only.py \
    --checkpoint new/models/phase4/checkpoints/best.pt \
    --data-dir new/data/sliding_multiscale \
    --hidden-dim 384 \
    --depth 10 \
    --batch-size 128 \
    2>&1 | tee new/models/phase4/logs/test.log

echo ""
echo "======================================================================"
echo "🎉 Phase 4 완료!"
echo "======================================================================"
echo ""
echo "📊 전체 Phase 비교:"
echo "  Phase 1: new/models/phase1/logs/test.log"
echo "  Phase 2: new/models/phase2/logs/test.log"
echo "  Phase 3: new/models/phase3/logs/test.log"
echo "  Phase 4: new/models/phase4/logs/test.log"
echo ""
echo "🎯 최종 목표 달성 여부 확인:"
echo "  - RMSE ≤ 2.0m"
echo "  - Within 1m ≥ 80%"
echo "  - Outliers >5m ≤ 1%"
echo ""
