#!/usr/bin/env python3
"""데이터 정리: additional_data + good bad files → data/raw/ (경로별 4개씩)"""
from pathlib import Path
from collections import defaultdict
import shutil

def main():
    # 소스 디렉토리
    raw_dir = Path("data/raw")
    bad_dir = Path("data/bad")
    additional_dir = Path("additional_data")

    # Good bad files 리스트
    good_bad_list = Path("analysis/outputs/good_bad_files.txt")

    # 1. 모든 파일 수집
    print("=" * 100)
    print("📊 데이터 정리 시작")
    print("=" * 100)
    print()

    all_files = defaultdict(list)

    # 1-1. 현재 raw 파일들
    print("1️⃣  현재 data/raw/ 파일 수집...")
    for csv_file in raw_dir.glob("*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            path = f"{parts[0]}->{parts[1]}"
            all_files[path].append(("raw", csv_file))

    raw_count = sum(len(files) for files in all_files.values())
    print(f"   ✅ {raw_count}개 파일")

    # 1-2. additional_data 파일들
    print("\n2️⃣  additional_data/ 파일 수집...")
    additional_count = 0
    for csv_file in additional_dir.glob("*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            path = f"{parts[0]}->{parts[1]}"
            all_files[path].append(("additional", csv_file))
            additional_count += 1

    print(f"   ✅ {additional_count}개 파일")

    # 1-3. Good bad 파일들
    print("\n3️⃣  Good bad 파일 수집...")
    good_bad_files = []
    with good_bad_list.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                good_bad_files.append(line)

    good_bad_count = 0
    for filename in good_bad_files:
        csv_file = bad_dir / filename
        if csv_file.exists():
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                path = f"{parts[0]}->{parts[1]}"
                all_files[path].append(("bad", csv_file))
                good_bad_count += 1

    print(f"   ✅ {good_bad_count}개 파일")

    # 2. 경로별 통계
    print("\n" + "=" * 100)
    print("📊 경로별 샘플 수")
    print("=" * 100)
    print()

    path_stats = []
    for path, files in sorted(all_files.items()):
        raw_files = [f for src, f in files if src == "raw"]
        additional_files = [f for src, f in files if src == "additional"]
        bad_files = [f for src, f in files if src == "bad"]

        path_stats.append({
            "path": path,
            "raw": len(raw_files),
            "additional": len(additional_files),
            "bad": len(bad_files),
            "total": len(files)
        })

    print(f"{'경로':<12} {'Raw':<6} {'Add':<6} {'Bad':<6} {'합계':<6} {'상태':<10}")
    print("-" * 100)

    for stat in sorted(path_stats, key=lambda x: x["total"], reverse=True):
        status = "✅ 사용" if stat["total"] >= 4 else f"❌ 부족 ({stat['total']}개)"
        print(f"{stat['path']:<12} {stat['raw']:<6} {stat['additional']:<6} "
              f"{stat['bad']:<6} {stat['total']:<6} {status:<10}")

    # 3. 4개 이상인 경로 선택
    usable_paths = {stat["path"]: stat for stat in path_stats if stat["total"] >= 4}

    print(f"\n✅ 사용 가능 경로: {len(usable_paths)}개")
    print(f"❌ 제외 경로 (샘플 < 4): {len(path_stats) - len(usable_paths)}개")

    # 4. 파일 정리 계획
    print("\n" + "=" * 100)
    print("📝 파일 정리 계획")
    print("=" * 100)
    print()

    final_files = []
    unused_files = []  # 사용하지 않는 파일들

    for path in sorted(usable_paths.keys()):
        files = all_files[path]
        total = len(files)

        # 목표 개수 결정
        if total == 5:
            target = 5  # 5개면 5개 사용
        elif total >= 6:
            target = 5  # 6개 이상이면 5개만 사용
        else:  # total == 4
            target = 4  # 4개면 4개 사용

        # 우선순위: raw > additional > bad
        selected = []

        # 1순위: raw
        raw_files = [(src, f) for src, f in files if src == "raw"]
        selected.extend(raw_files[:min(target, len(raw_files))])

        # 2순위: additional
        if len(selected) < target:
            additional_files = [(src, f) for src, f in files if src == "additional"]
            need = target - len(selected)
            selected.extend(additional_files[:need])

        # 3순위: bad
        if len(selected) < target:
            bad_files = [(src, f) for src, f in files if src == "bad"]
            need = target - len(selected)
            selected.extend(bad_files[:need])

        # 정확히 target개만
        selected = selected[:target]

        # 사용하지 않는 파일들 기록
        all_file_paths = [f for _, f in files]
        selected_paths = [f for _, f in selected]
        for f in all_file_paths:
            if f not in selected_paths:
                unused_files.append(f)

        print(f"{path:<12}: {len(selected)}개 선택 (전체 {total}개)")
        for idx, (src, file_path) in enumerate(selected, 1):
            # 파일명 정리: {start}_{end}_{idx}.csv
            start, end = path.split("->")
            new_name = f"{start}_{end}_{idx}.csv"
            final_files.append((file_path, new_name, src))

    # 제외된 경로의 파일들도 unused에 추가
    for path, files in all_files.items():
        if path not in usable_paths:
            for _, f in files:
                unused_files.append(f)

    print(f"\n총 {len(final_files)}개 파일 → data/raw/")

    # 5. 사용자 확인
    print("\n" + "=" * 100)
    print("⚠️  확인")
    print("=" * 100)
    print(f"""
현재 data/raw/: {raw_count}개 파일
새로 정리: {len(final_files)}개 파일 ({len(usable_paths)}개 경로 × 4개)

추가:
  - additional_data: {sum(1 for _, _, src in final_files if src == 'additional')}개
  - good bad files: {sum(1 for _, _, src in final_files if src == 'bad')}개

⚠️  기존 data/raw/ 파일은 data/raw_backup/으로 백업됩니다.
""")

    response = input("계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("❌ 취소되었습니다.")
        return

    # 6. 백업 및 정리
    print("\n" + "=" * 100)
    print("🔄 파일 정리 중...")
    print("=" * 100)

    # 백업
    backup_dir = Path("data/raw_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    for old_file in raw_dir.glob("*.csv"):
        shutil.move(str(old_file), str(backup_dir / old_file.name))

    print(f"✅ 기존 파일 백업: data/raw_backup/ ({raw_count}개)")

    # 새 파일 복사
    for src_path, new_name, source_type in final_files:
        dest_path = raw_dir / new_name
        # raw 소스는 백업에서 복사
        if source_type == "raw":
            src_path = backup_dir / src_path.name
        shutil.copy2(str(src_path), str(dest_path))

    print(f"✅ 새 파일 복사: data/raw/ ({len(final_files)}개)")

    # Unused 파일들을 data/unused/로 이동
    unused_dir = Path("data/unused")
    if unused_dir.exists():
        shutil.rmtree(unused_dir)
    unused_dir.mkdir(parents=True, exist_ok=True)

    for unused_file in unused_files:
        dest_path = unused_dir / unused_file.name
        # bad 폴더에서 온 파일만 이동 (raw/additional은 백업에 있음)
        if unused_file.parent == bad_dir:
            shutil.move(str(unused_file), str(dest_path))

    # 제외된 경로의 bad 파일들도 이동
    excluded_bad_count = sum(1 for f in unused_files if f.parent == bad_dir)
    print(f"✅ 미사용 파일 이동: data/unused/ ({excluded_bad_count}개)")

    # 7. 최종 통계
    print("\n" + "=" * 100)
    print("📊 최종 결과")
    print("=" * 100)

    final_stats = defaultdict(int)
    for csv_file in raw_dir.glob("*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            path = f"{parts[0]}->{parts[1]}"
            final_stats[path] += 1

    print(f"\n경로 수: {len(final_stats)}개")
    print(f"총 파일: {sum(final_stats.values())}개")
    print(f"경로당 평균: {sum(final_stats.values()) / len(final_stats):.1f}개")

    # 경로별 확인
    print(f"\n경로별 파일 수:")
    for path in sorted(final_stats.keys()):
        count = final_stats[path]
        status = "✅" if count == 4 else f"⚠️  {count}개"
        print(f"  {path:<12}: {status}")

    print("\n" + "=" * 100)
    print("✅ 완료!")
    print("=" * 100)
    print(f"""
백업: data/raw_backup/
새 데이터: data/raw/ ({len(final_files)}개 파일, {len(usable_paths)}개 경로)

다음 단계:
  python src/preprocess_sliding.py --output data/sliding_mag4_adaptive
""")

if __name__ == "__main__":
    main()
