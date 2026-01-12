# Analysis Scripts 가이드

## 📂 분석 코드 분류

### 1️⃣ Outlier 분석 (5개)

**목적:** 큰 오차(3m 이상) 샘플 분석 및 원인 규명

| 파일 | 설명 | 출력 |
|------|------|------|
| `analyze_outlier_cause.py` | **핵심 분석** - Outlier 원인 심층 분석 | X/Y 방향 오차, 자기장 변동성 |
| `analyze_outliers.py` | 기본 Outlier 통계 | 개수, 비율, 분포 |
| `deep_analyze_outliers.py` | 심층 Outlier 패턴 분석 | 시각화 + 상세 통계 |
| `find_outlier_patterns.py` | Outlier 공통 패턴 탐색 | 패턴 목록 |
| `find_outlier_windows.py` | Outlier 발생 윈도우 특정 | 윈도우 인덱스 |

**실행 순서:** outlier_cause → outliers → deep_outliers

---

### 2️⃣ 데이터 품질 분석 (4개)

**목적:** 불량 데이터 식별 및 품질 평가

| 파일 | 설명 | 출력 |
|------|------|------|
| `analyze_file_quality.py` | **핵심** - 파일별 품질 점수 | 파일별 평가 |
| `analyze_bad_data.py` | 불량 데이터 특성 분석 | 불량 샘플 통계 |
| `deep_analyze_bad.py` | 불량 데이터 심층 분석 | 상세 원인 |
| `move_good_bad_to_raw.py` | 좋은/나쁜 데이터 분리 | 파일 이동 |

**실행 순서:** file_quality → bad_data → move_good_bad

---

### 3️⃣ 데이터 분포 분석 (3개)

**목적:** 데이터셋 균형 및 경로별 특성 분석

| 파일 | 설명 | 출력 |
|------|------|------|
| `analyze_data_imbalance.py` | **핵심** - Train/Val/Test 균형 분석 | 분할 통계, 시각화 |
| `quick_imbalance_check.py` | 빠른 불균형 체크 | 간단한 통계 |
| `analyze_per_path.py` | 경로별 Pitch/Roll/Yaw 분석 | 경로별 특성 |

**실행 순서:** imbalance → per_path

---

### 4️⃣ 모델 성능 분석 (3개)

**목적:** 모델 강건성 및 특성 평가

| 파일 | 설명 | 출력 |
|------|------|------|
| `analyze_noise_robustness.py` | **핵심** - 노이즈 강건성 테스트 | 노이즈 레벨별 성능 |
| `analyze_calibration_cause.py` | 캘리브레이션 drift 분석 | Drift 원인 |
| `analyze_for_hyena.py` | Hyena 모델 특화 분석 | 모델 적합성 |

**실행 순서:** noise_robustness → calibration

---

### 5️⃣ 기본 분석 (3개)

**목적:** 전반적인 데이터 이해

| 파일 | 설명 | 출력 |
|------|------|------|
| `fundamental_analysis.py` | 기본 통계 분석 | 전체 데이터 통계 |
| `analyze_orientation.py` | 방향 센서 분석 | Pitch/Roll/Yaw 패턴 |
| `visualize_features.py` | Feature 시각화 | 자기장, 각도 그래프 |

---

## 📊 분석 결과물 (outputs/)

```
outputs/
├── deep_outlier_analysis.png      # Outlier 시각화
├── noise_robustness_analysis.png  # 노이즈 강건성 그래프
├── split_comparison.png           # Train/Val/Test 비교
├── feature_analysis_1_11_1.png    # Feature 시각화
├── exclude_files.txt              # 제외할 파일 목록
├── good_bad_files.txt             # 품질별 파일 분류
└── raw_style_bad_files.txt        # Raw 형식 불량 파일
```

---

## 🚀 사용 가이드

### 전체 분석 실행 순서

```bash
cd analysis

# 1. 기본 분석
python fundamental_analysis.py
python visualize_features.py

# 2. 데이터 품질
python analyze_file_quality.py
python analyze_bad_data.py

# 3. 데이터 분포
python analyze_data_imbalance.py
python analyze_per_path.py

# 4. Outlier 분석
python analyze_outlier_cause.py
python deep_analyze_outliers.py

# 5. 모델 성능
python analyze_noise_robustness.py
python analyze_calibration_cause.py
```

### 주요 분석 스크립트만 실행

```bash
# 필수 4개
python analyze_outlier_cause.py       # Outlier 원인
python analyze_noise_robustness.py    # 노이즈 강건성
python analyze_data_imbalance.py      # 데이터 균형
python analyze_file_quality.py        # 파일 품질
```

---

## 📁 디렉토리 구조 제안

현재는 flat 구조지만, 카테고리별로 정리하면:

```
analysis/
├── outliers/
│   ├── analyze_outlier_cause.py
│   ├── analyze_outliers.py
│   ├── deep_analyze_outliers.py
│   ├── find_outlier_patterns.py
│   └── find_outlier_windows.py
├── quality/
│   ├── analyze_file_quality.py
│   ├── analyze_bad_data.py
│   ├── deep_analyze_bad.py
│   └── move_good_bad_to_raw.py
├── distribution/
│   ├── analyze_data_imbalance.py
│   ├── quick_imbalance_check.py
│   └── analyze_per_path.py
├── performance/
│   ├── analyze_noise_robustness.py
│   ├── analyze_calibration_cause.py
│   └── analyze_for_hyena.py
├── basic/
│   ├── fundamental_analysis.py
│   ├── analyze_orientation.py
│   └── visualize_features.py
├── outputs/
└── README.md (이 파일)
```

디렉토리 정리할까요?

---

## 🎯 분석 목적별 사용법

### 성능 개선하고 싶을 때
1. `analyze_outlier_cause.py` - 어디서 오차 큰지
2. `analyze_noise_robustness.py` - 노이즈 영향
3. `analyze_data_imbalance.py` - 데이터 불균형

### 데이터 문제 찾고 싶을 때
1. `analyze_file_quality.py` - 어떤 파일이 문제?
2. `analyze_bad_data.py` - 왜 나쁜지?
3. `move_good_bad_to_raw.py` - 분리하기

### 모델 이해하고 싶을 때
1. `fundamental_analysis.py` - 전체 통계
2. `visualize_features.py` - 데이터 시각화
3. `analyze_for_hyena.py` - Hyena 특성

---

**Last Updated**: 2025-11-26
**Total Scripts**: 18개
