# 🚀 Phase별 학습 자동화 스크립트

딸깍 한 번에 전체 파이프라인 실행!

---

## 📁 파일 구조

```
scripts/
├── run_phase1.sh          # Phase 1: Baseline (최적화된 파라미터)
├── run_phase2.sh          # Phase 2: Gradient Features 추가
├── run_phase3.sh          # Phase 3: Outlier Removal
├── run_phase4.sh          # Phase 4: Attention + Multi-scale
├── RESULT_GUIDE.md        # 결과 분석 및 의사결정 가이드
├── ERROR_GUIDE.md         # 에러 발생 시 디버깅 가이드
└── README.md              # 이 파일
```

---

## 🎯 빠른 시작

### Phase 1: Baseline 학습

```bash
cd /Users/yunho/school/lstm
./scripts/run_phase1.sh
```

**소요 시간**: 8-10시간
**결과**: `new/models/phase1/checkpoints/best.pt`

### Phase 2: Gradient Features

```bash
./scripts/run_phase2.sh
```

**소요 시간**: 8-10시간
**결과**: `new/models/phase2/checkpoints/best.pt`

### Phase 3: Outlier Removal

```bash
./scripts/run_phase3.sh
```

**소요 시간**: 8-10시간
**결과**: `new/models/phase3/checkpoints/best.pt`

### Phase 4: Attention + Multi-scale

```bash
./scripts/run_phase4.sh
```

**소요 시간**: 10-12시간
**결과**: `new/models/phase4/checkpoints/best.pt`

---

## 📊 각 Phase 설정

| Phase | Features | 특징 | 예상 RMSE |
|-------|----------|------|-----------|
| **1** | 4개 (MagX/Y/Z, Mag) | 최적화된 학습 파라미터 | 2.3-2.5m |
| **2** | 8개 (+ Gradient) | 변화율 정보 추가 | 2.0-2.2m |
| **3** | 8개 | Outlier 제거 후 재학습 | 1.8-2.0m |
| **4** | 8개 | Multi-scale + Attention | 1.7-1.9m |

**공통 설정**:
- LR: 1e-4
- Warmup: 10 epochs
- Total Epochs: 400
- Patience: 20 (Phase 3-4는 25)
- Best Model: P90 기준
- LR Scheduler: RMSE 기준

---

## 🔍 결과 확인

### 테스트 결과 보기

```bash
# Phase 1 결과
cat new/models/phase1/logs/test.log

# Phase 2 결과
cat new/models/phase2/logs/test.log

# 비교
diff <(grep "RMSE" new/models/phase1/logs/test.log) \
     <(grep "RMSE" new/models/phase2/logs/test.log)
```

### 학습 로그 보기

```bash
# 학습 진행 상황
tail -f new/models/phase1/logs/train.log

# Best epoch 확인
grep "Best" new/models/phase1/logs/train.log
```

---

## 📋 다음 단계 결정

각 Phase 완료 후, **RESULT_GUIDE.md**를 참고하여:

1. **결과 템플릿 작성**
2. **의사결정** (다음 Phase 진행 or 재조정)
3. **파라미터 조정** (필요시)

### 예시: Phase 2 완료 후

```bash
# 1. 결과 확인
cat new/models/phase2/logs/test.log

# 2. Phase 1과 비교
echo "=== Phase 1 ==="
grep "RMSE" new/models/phase1/logs/test.log
echo "=== Phase 2 ==="
grep "RMSE" new/models/phase2/logs/test.log

# 3. RESULT_GUIDE.md의 템플릿 작성
# 4. 개선폭 확인 → Phase 3 진행 여부 결정
```

---

## ⚠️ 에러 발생 시

**ERROR_GUIDE.md** 참고:

1. **에러 메시지 확인**
2. **로그 파일 확인**
3. **템플릿 작성**
4. **질문하기**

### 빠른 디버깅

```bash
# 학습 로그 마지막 50줄
tail -50 new/models/phase1/logs/train.log

# 전체 로그 보기
less new/models/phase1/logs/train.log
```

---

## 🎨 커스터마이징

### 파라미터 수정

스크립트 내부에서 직접 수정 가능:

```bash
# run_phase1.sh 편집
nano scripts/run_phase1.sh

# 예: Batch size 변경
--batch-size 64  # 128 → 64

# 예: Learning rate 변경
--lr 5e-5  # 1e-4 → 5e-5

# 예: Depth 변경
--depth 12  # 10 → 12
```

### 중간 중단 및 재개

```bash
# Ctrl+C로 중단 가능 (Best model은 이미 저장됨)

# 재개: 동일 스크립트 다시 실행
./scripts/run_phase1.sh
# → 체크포인트 있으면 이어서 학습 가능 (현재 미구현)
```

---

## 📈 전체 진행 상황 추적

### 전체 결과 요약

```bash
# 모든 Phase 결과 한눈에
for phase in 1 2 3 4; do
    echo "=== Phase $phase ==="
    if [ -f "new/models/phase$phase/logs/test.log" ]; then
        grep -A 2 "MAE:" new/models/phase$phase/logs/test.log | head -3
    else
        echo "  (아직 실행 안 됨)"
    fi
    echo ""
done
```

### 시간 추적

```bash
# 각 Phase 시작/종료 시간
ls -lh new/models/phase*/checkpoints/best.pt

# 학습 시간 확인
head -1 new/models/phase1/logs/train.log  # 시작 시간
tail -5 new/models/phase1/logs/train.log  # 종료 시간
```

---

## 🎯 목표 달성 확인

### 최종 목표

- ✅ RMSE ≤ 2.0m
- ✅ Within 1m ≥ 80%
- ✅ Max error ≤ 5m

### 확인 방법

```bash
# Phase 3 또는 4 결과 확인
cat new/models/phase3/logs/test.log | grep -E "RMSE|Within 1m|Max"

# 예시 출력:
# RMSE: 1.9m  ✅
# Within 1m: 82%  ✅
# Max: 4.8m  ✅
```

---

## 💡 Tips

### 1. 백그라운드 실행

```bash
# 터미널 꺼도 계속 실행
nohup ./scripts/run_phase1.sh > phase1.out 2>&1 &

# 진행 상황 확인
tail -f phase1.out
```

### 2. 여러 Phase 순차 실행

```bash
# Phase 1-2-3 자동 실행
./scripts/run_phase1.sh && \
./scripts/run_phase2.sh && \
./scripts/run_phase3.sh
```

### 3. 결과 알림

```bash
# Phase 완료 시 알림 (macOS)
./scripts/run_phase1.sh && osascript -e 'display notification "Phase 1 완료!" with title "학습 완료"'
```

### 4. 디스크 용량 확인

```bash
# 모델 파일 크기
du -sh new/models/phase*/checkpoints/

# 로그 파일 크기
du -sh new/models/phase*/logs/
```

---

## 🔧 고급 사용법

### Phase별 선택 실행

```bash
# Phase 1 skip하고 Phase 2부터
# (Phase 1 모델 이미 있을 때)
./scripts/run_phase2.sh

# Phase 3만 재실행
rm -rf new/models/phase3/
./scripts/run_phase3.sh
```

### Outlier Threshold 변경

```bash
# run_phase3.sh 수정
nano scripts/run_phase3.sh

# Line 찾기: --threshold 5.0
# 변경: --threshold 4.5  # 더 공격적
# 또는: --threshold 5.5  # 더 보수적
```

### 전처리 재실행 강제

```bash
# 전처리 캐시 삭제
rm new/data/sliding_mag4/meta.json
rm new/data/sliding_grad/meta.json

# 스크립트 재실행 → 전처리부터 다시
./scripts/run_phase1.sh
```

---

## 📞 도움말

- **결과 분석**: `RESULT_GUIDE.md` 참고
- **에러 해결**: `ERROR_GUIDE.md` 참고
- **파라미터 설명**: `new/README.md` 참고

---

**Good Luck! 🚀**
