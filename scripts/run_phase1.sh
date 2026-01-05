#!/usr/bin/env bash
# Phase 1: Baseline (최적화된 학습 파라미터)
# 딸깍 한 번에 전체 파이프라인 실행

set -e  # 에러 발생 시 중단

PROJECT_ROOT="/Users/yunho/school/lstm"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Phase 1: Baseline Training"
echo "======================================================================"
echo ""
echo "📋 설정:"
echo "  - Features: 4 (MagX, MagY, MagZ, Magnitude)"
echo "  - LR: 1e-4"
echo "  - Warmup: 10 epochs"
echo "  - Total Epochs: 400"
echo "  - Patience: 20"
echo "  - Best Model: P90 기준"
echo "  - LR Scheduler: RMSE 기준"
echo ""
echo "======================================================================"

# 1. 가상환경 활성화
echo ""
echo "✅ Step 1: 가상환경 활성화"
source venv/bin/activate

# 2. 전처리 (이미 있으면 skip)
echo ""
echo "✅ Step 2: 전처리 확인"
if [ -f "new/data/sliding_mag4/meta.json" ]; then
    echo "   ℹ️  전처리 데이터 이미 존재 → Skip"
else
    echo "   🔄 전처리 시작..."
    python new/src/preprocess_from_csv.py \
        --raw-dir data/raw \
        --nodes-file new/data/nodes_final.csv \
        --output-dir new/data/preprocessed
    echo "   ✅ 전처리 완료"
fi

# 3. 출력 디렉토리 생성
echo ""
echo "✅ Step 3: 출력 디렉토리 생성"
mkdir -p new/models/phase1/checkpoints
mkdir -p new/models/phase1/logs

# 4. 학습 시작
echo ""
echo "======================================================================"
echo "🔥 Step 4: 학습 시작 (예상 시간: 8-10시간)"
echo "======================================================================"
echo ""

python new/src/train_sliding.py \
    --data-dir new/data/sliding_mag4 \
    --epochs 400 \
    --batch-size 128 \
    --lr 1e-4 \
    --warmup-epochs 10 \
    --patience 20 \
    --hidden-dim 384 \
    --depth 10 \
    --checkpoint-dir new/models/phase1/checkpoints \
    2>&1 | tee new/models/phase1/logs/train.log

echo ""
echo "======================================================================"
echo "✅ 학습 완료!"
echo "======================================================================"

# 5. 테스트
echo ""
echo "✅ Step 5: 테스트"
python new/src/test_only.py \
    --checkpoint new/models/phase1/checkpoints/best.pt \
    --data-dir new/data/sliding_mag4 \
    --hidden-dim 384 \
    --depth 10 \
    --batch-size 128 \
    2>&1 | tee new/models/phase1/logs/test.log

echo ""
echo "======================================================================"
echo "🎉 Phase 1 완료!"
echo "======================================================================"
echo ""
echo "📁 결과 위치:"
echo "  - 모델: new/models/phase1/checkpoints/best.pt"
echo "  - 로그: new/models/phase1/logs/"
echo ""
echo "📊 다음 단계:"
echo "  ./scripts/run_phase2.sh  # Gradient Features 추가"
echo ""
