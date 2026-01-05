#!/usr/bin/env python3
"""센서 데이터와 위치의 상관관계 시각화"""
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
try:
    import platform
    system = platform.system()
    if system == "Darwin":  # macOS
        plt.rc("font", family="AppleGothic")
    elif system == "Windows":
        plt.rc("font", family="Malgun Gothic")
    else:  # Linux
        plt.rc("font", family="NanumGothic")
    plt.rc("axes", unicode_minus=False)
except Exception:
    print("⚠️  한국어 폰트 설정 실패")

# nodes_final.csv 읽기
nodes_path = Path("data/nodes_final.csv")
positions = {}
with nodes_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        positions[int(row["id"])] = (float(row["x_m"]), float(row["y_m"]))

# CSV 파일 선택
if len(sys.argv) > 1:
    csv_file = Path(sys.argv[1])
else:
    # 기본 파일
    csv_file = Path("data/raw/1_11_1.csv")

if not csv_file.exists():
    print(f"파일 없음: {csv_file}")
    sys.exit(1)

print(f"📊 분석 파일: {csv_file.name}")

# 데이터 읽기
timestamps = []
magx_vals = []
magy_vals = []
magz_vals = []
pitch_vals = []
roll_vals = []
yaw_vals = []

with csv_file.open() as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        timestamps.append(i)
        magx_vals.append(float(row["MagX"]))
        magy_vals.append(float(row["MagY"]))
        magz_vals.append(float(row["MagZ"]))
        pitch_vals.append(float(row["Pitch"]))
        roll_vals.append(float(row["Roll"]))
        yaw_vals.append(float(row["Yaw"]))

# 경로 정보
start_node, end_node = map(int, csv_file.stem.split("_")[:2])
start_pos = positions[start_node]
end_pos = positions[end_node]

print(f"  경로: {start_node} → {end_node}")
print(f"  시작: {start_pos}, 종료: {end_pos}")
print(f"  데이터: {len(timestamps)}개")

# 시각화
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# 1. 지자기 3축
ax = axes[0, 0]
ax.plot(timestamps, magx_vals, label="MagX", alpha=0.7)
ax.plot(timestamps, magy_vals, label="MagY", alpha=0.7)
ax.plot(timestamps, magz_vals, label="MagZ", alpha=0.7)
ax.set_xlabel("타임스텝")
ax.set_ylabel("지자기 (μT)")
ax.set_title(f"지자기 센서 (경로: {start_node}→{end_node})")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Pitch
ax = axes[0, 1]
ax.plot(timestamps, pitch_vals, color="blue", alpha=0.7)
ax.axhline(y=sum(pitch_vals)/len(pitch_vals), color="red", linestyle="--", label=f"평균={sum(pitch_vals)/len(pitch_vals):.1f}°")
ax.set_xlabel("타임스텝")
ax.set_ylabel("Pitch (도)")
ax.set_title(f"Pitch 변화 (std={sum((x - sum(pitch_vals)/len(pitch_vals))**2 for x in pitch_vals)**0.5/len(pitch_vals)**0.5:.2f}°)")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Roll
ax = axes[1, 0]
ax.plot(timestamps, roll_vals, color="green", alpha=0.7)
ax.axhline(y=sum(roll_vals)/len(roll_vals), color="red", linestyle="--", label=f"평균={sum(roll_vals)/len(roll_vals):.1f}°")
ax.set_xlabel("타임스텝")
ax.set_ylabel("Roll (도)")
ax.set_title(f"Roll 변화 (std={sum((x - sum(roll_vals)/len(roll_vals))**2 for x in roll_vals)**0.5/len(roll_vals)**0.5:.2f}°)")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Yaw
ax = axes[1, 1]
ax.plot(timestamps, yaw_vals, color="orange", alpha=0.7)
ax.set_xlabel("타임스텝")
ax.set_ylabel("Yaw (도)")
ax.set_title("Yaw 변화 (방향)")
ax.grid(True, alpha=0.3)

# 5. MagX vs MagY (평면)
ax = axes[2, 0]
sc = ax.scatter(magx_vals, magy_vals, c=timestamps, cmap="viridis", alpha=0.6, s=10)
ax.set_xlabel("MagX (μT)")
ax.set_ylabel("MagY (μT)")
ax.set_title("지자기 평면 (MagX-MagY) - 시간 순서")
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label="타임스텝")

# 6. 지자기 크기
ax = axes[2, 1]
mag_magnitude = [(x**2 + y**2 + z**2)**0.5 for x, y, z in zip(magx_vals, magy_vals, magz_vals)]
ax.plot(timestamps, mag_magnitude, color="purple", alpha=0.7)
ax.set_xlabel("타임스텝")
ax.set_ylabel("지자기 크기 (μT)")
ax.set_title("지자기 벡터 크기 (Magnitude)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path(f"analysis/outputs/feature_analysis_{csv_file.stem}.png")
plt.savefig(output_path, dpi=150)
print(f"\n✅ 저장: {output_path}")

plt.show()
