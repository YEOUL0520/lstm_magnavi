#!/bin/bash
# 전처리된 CSV → JSONL 변환

set -e  # 에러 발생시 중단

# 파라미터 받기 (기본값 설정)
WINDOW_SIZE=${1:-250}
STRIDE=${2:-25}

echo "=========================================="
echo "CSV → JSONL 변환 시작"
echo "=========================================="
echo "Window Size: $WINDOW_SIZE"
echo "Stride:      $STRIDE"
echo ""

# preprocess_from_csv.py는 기본값을 사용하므로 필요시 수정 필요
# 현재는 window_size=250, stride=25 하드코딩
python3 src/preprocess_from_csv.py

echo ""
echo "✅ 전처리 완료!"
