#!/usr/bin/env bash
# Phase 3: Outlier Removal
# Phase 2 모델로 outlier 분석 → 5m 이상 제거 → 재학습

set -e

PROJECT_ROOT="/Users/yunho/school/lstm"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Phase 3: Outlier Removal"
echo "======================================================================"
echo ""
echo "📋 설정:"
echo "  - Phase 2 모델로 outlier 분석"
echo "  - Threshold: 5m 이상 제거"
echo "  - 필터링된 데이터로 재학습"
echo ""
echo "======================================================================"

# 1. 가상환경 활성화
echo ""
echo "✅ Step 1: 가상환경 활성화"
source venv/bin/activate

# 2. Phase 2 모델 확인
echo ""
echo "✅ Step 2: Phase 2 모델 확인"
if [ ! -f "new/models/phase2/checkpoints/best.pt" ]; then
    echo "❌ 에러: Phase 2 모델이 없습니다!"
    echo "   먼저 ./scripts/run_phase2.sh 를 실행하세요."
    exit 1
fi
echo "   ✅ Phase 2 모델 존재"

# 3. Outlier 분석
echo ""
echo "✅ Step 3: Outlier 분석"
mkdir -p analysis/outputs

python scripts/analyze_and_filter_outliers.py \
    --checkpoint new/models/phase2/checkpoints/best.pt \
    --data-dir new/data/sliding_grad \
    --hidden-dim 384 \
    --depth 10 \
    2>&1 | tee analysis/outputs/outlier_analysis_phase2.log

echo ""
echo "   📊 분석 결과: analysis/outputs/outlier_analysis.json"

# 4. Outlier 필터링
echo ""
echo "✅ Step 4: Outlier 필터링 (threshold=5.0m)"
python scripts/analyze_and_filter_outliers.py \
    --checkpoint new/models/phase2/checkpoints/best.pt \
    --data-dir new/data/sliding_grad \
    --hidden-dim 384 \
    --depth 10 \
    --filter \
    --threshold 5.0 \
    --output-dir new/data/sliding_grad_filtered \
    2>&1 | tee analysis/outputs/outlier_filter.log

echo "   ✅ 필터링 완료: new/data/sliding_grad_filtered/"

# 5. 출력 디렉토리 생성
echo ""
echo "✅ Step 5: 출력 디렉토리 생성"
mkdir -p new/models/phase3/checkpoints
mkdir -p new/models/phase3/logs

# 6. 재학습 (필터링된 데이터)
echo ""
echo "======================================================================"
echo "🔥 Step 6: 재학습 (Outlier 제거된 데이터)"
echo "======================================================================"
echo ""

python new/src/train_sliding.py \
    --data-dir new/data/sliding_grad_filtered \
    --epochs 400 \
    --batch-size 128 \
    --lr 1e-4 \
    --warmup-epochs 10 \
    --patience 25 \
    --hidden-dim 384 \
    --depth 10 \
    --checkpoint-dir new/models/phase3/checkpoints \
    2>&1 | tee new/models/phase3/logs/train.log

echo ""
echo "======================================================================"
echo "✅ 학습 완료!"
echo "======================================================================"

# 7. 테스트
echo ""
echo "✅ Step 7: 테스트"
python new/src/test_only.py \
    --checkpoint new/models/phase3/checkpoints/best.pt \
    --data-dir new/data/sliding_grad_filtered \
    --hidden-dim 384 \
    --depth 10 \
    --batch-size 128 \
    2>&1 | tee new/models/phase3/logs/test.log

echo ""
echo "======================================================================"
echo "🎉 Phase 3 완료!"
echo "======================================================================"
echo ""
echo "📊 결과:"
echo "  Phase 2: new/models/phase2/logs/test.log"
echo "  Phase 3: new/models/phase3/logs/test.log"
echo "  Outlier 분석: analysis/outputs/outlier_analysis.json"
echo ""
echo "📁 다음 단계:"
echo "  ./scripts/run_phase4.sh  # Attention + Multi-scale"
echo ""
