#!/bin/bash
# 가상환경 생성 및 패키지 설치 스크립트

set -e  # 에러 발생 시 중단

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

echo "🐍 Python 가상환경 설정 시작..."
echo ""

# 가상환경 존재 여부 확인
if [ -d "venv" ]; then
    echo "⚠️  기존 가상환경 발견. 삭제하고 새로 만들까요? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  기존 가상환경 삭제 중..."
        rm -rf venv
    else
        echo "❌ 취소됨. 기존 가상환경을 사용하려면 'source venv/bin/activate'를 실행하세요."
        exit 0
    fi
fi

# 1. 가상환경 생성
echo "📦 가상환경 생성 중..."
python3 -m venv venv

# 2. 가상환경 활성화
echo "✅ 가상환경 생성 완료"
echo ""
echo "🔧 패키지 설치 중..."
source venv/bin/activate

# 3. pip 업그레이드
pip install --upgrade pip

# 4. 패키지 설치
pip install -r requirements.txt

echo ""
echo "================================"
echo "✅ 설치 완료!"
echo "================================"
echo ""
echo "📝 다음 명령어로 가상환경을 활성화하세요:"
echo "   source venv/bin/activate"
echo ""
echo "🚀 그 다음 실행:"
echo "   ./scripts/run_all.sh"
echo ""
