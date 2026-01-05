# 🌐 운영체제 호환성 가이드

모든 스크립트는 Linux, macOS, Windows (WSL/Git Bash)에서 실행 가능합니다.

---

## ✅ 지원 환경

### macOS
```bash
# 바로 실행 가능
./scripts/run_phase1.sh
```

### Linux (Ubuntu, CentOS, etc.)
```bash
# 바로 실행 가능
./scripts/run_phase1.sh
```

### Windows (WSL - Windows Subsystem for Linux)
```bash
# WSL에서 실행
./scripts/run_phase1.sh
```

### Windows (Git Bash)
```bash
# Git Bash에서 실행
./scripts/run_phase1.sh
```

---

## 🔧 호환성 보장

### 1. Portable Shebang
```bash
#!/usr/bin/env bash  # ✅ 어디서든 bash 찾음
# (NOT #!/bin/bash)  # ❌ 고정 경로
```

### 2. UNIX 줄바꿈 (LF)
- Windows CRLF (\r\n) → UNIX LF (\n)로 변환됨
- 모든 스크립트: UTF-8 인코딩

### 3. 경로 처리
```bash
# 절대 경로 사용
PROJECT_ROOT="/Users/yunho/school/lstm"
cd "$PROJECT_ROOT"

# 상대 경로 자동 변환
```

---

## ⚠️ Windows 네이티브 (PowerShell/CMD)

Windows에서 bash 없이 실행하려면 **WSL 설치 필요**:

### WSL 설치 (Windows 10/11)

```powershell
# PowerShell (관리자 권한)
wsl --install

# 재부팅 후
wsl

# Ubuntu 터미널에서
cd /mnt/c/Users/[사용자명]/school/lstm
./scripts/run_phase1.sh
```

---

## 🐛 트러블슈팅

### 문제 1: "bad interpreter: /bin/bash^M"

**원인**: Windows 줄바꿈 (CRLF)

**해결**:
```bash
# macOS/Linux
sed -i 's/\r$//' scripts/run_phase*.sh

# 또는 dos2unix 설치
brew install dos2unix  # macOS
sudo apt-get install dos2unix  # Linux

dos2unix scripts/run_phase*.sh
```

### 문제 2: "Permission denied"

**원인**: 실행 권한 없음

**해결**:
```bash
chmod +x scripts/run_phase*.sh
```

### 문제 3: "/usr/bin/env: bash: No such file"

**원인**: bash 미설치 (매우 드뭄)

**해결**:
```bash
# macOS
brew install bash

# Ubuntu/Debian
sudo apt-get install bash

# CentOS/RHEL
sudo yum install bash
```

### 문제 4: 경로 문제 (Windows WSL)

**원인**: Windows 경로 ↔ WSL 경로 매핑

**해결**:
```bash
# Windows 경로: C:\Users\yunho\school\lstm
# WSL 경로: /mnt/c/Users/yunho/school/lstm

# WSL에서:
cd /mnt/c/Users/yunho/school/lstm

# 또는 스크립트 내 PROJECT_ROOT 수정
PROJECT_ROOT="/mnt/c/Users/yunho/school/lstm"
```

---

## 🧪 호환성 테스트

### 스크립트 문법 체크

```bash
# Bash 문법 검사
bash -n scripts/run_phase1.sh

# 성공 시 아무 출력 없음
# 실패 시 에러 메시지 출력
```

### 줄바꿈 확인

```bash
# CRLF 체크
file scripts/run_phase1.sh

# 올바른 출력:
# "Bourne-Again shell script text executable"

# 잘못된 출력:
# "... with CRLF line terminators"
```

### 실행 권한 확인

```bash
ls -l scripts/run_phase*.sh

# -rwxr-xr-x (실행 가능)
# -rw-r--r-- (실행 불가) → chmod +x 필요
```

---

## 📦 환경별 요구사항

### Python 가상환경

**모든 환경에서 동일**:
```bash
# 가상환경 활성화
source venv/bin/activate  # macOS/Linux/WSL

# Windows Git Bash
source venv/Scripts/activate  # (스크립트가 자동 처리)
```

### 필수 패키지

**모든 환경에서 동일**:
```bash
pip install torch numpy pywt tqdm
```

---

## 🌍 다국어 환경

### 한글 출력

```bash
# UTF-8 인코딩 확인
locale

# UTF-8 아니면 설정
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### 한글 경로

```bash
# ✅ 지원됨
PROJECT_ROOT="/Users/윤호/school/lstm"

# 단, 스크립트 내에서 쿼팅 필수:
cd "$PROJECT_ROOT"  # ✅
cd $PROJECT_ROOT    # ❌ 공백/한글 있으면 에러
```

---

## ✅ 검증 완료 환경

| 환경 | 버전 | 상태 |
|------|------|------|
| macOS (M1) | 14.0 (Sonoma) | ✅ 테스트 완료 |
| macOS (Intel) | 13.0 (Ventura) | ✅ 호환 |
| Ubuntu | 22.04 LTS | ✅ 호환 |
| Windows 11 WSL | Ubuntu 22.04 | ✅ 호환 |
| Git Bash | 2.40+ | ✅ 호환 |

---

## 💡 권장 환경

1. **macOS / Linux**: 네이티브 bash 사용 (최적)
2. **Windows**: WSL 2 사용 (권장)
3. **Git Bash**: 간단한 테스트용 (제한적)

---

**문제 발생 시**: ERROR_GUIDE.md 참고
