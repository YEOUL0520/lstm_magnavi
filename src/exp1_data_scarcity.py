#!/usr/bin/env python3
"""
Exp1) Data scarcity experiment
- Create reduced train.jsonl (keep val/test fixed)
- Output: exp_data/scarcity/{p}/data/sliding_lstm/train.jsonl + copied meta/val/test
"""

import json
import random
import shutil
from pathlib import Path
import argparse

def count_lines(p: Path) -> int:
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def sample_jsonl(in_path: Path, out_path: Path, ratio: float, seed: int = 42):
    random.seed(seed)
    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    n = len(lines)
    k = max(1, int(n * ratio))
    idx = list(range(n))
    random.shuffle(idx)
    pick = sorted(idx[:k])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in pick:
            f.write(lines[i])

    return n, k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data-dir", type=str, default="data/sliding_lstm")
    ap.add_argument("--out-root", type=str, default="exp_data/scarcity")
    ap.add_argument("--ratios", type=str, default="1.0,0.5,0.2,0.1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.base_data_dir)
    out_root = Path(args.out_root)

    train_in = base / "train.jsonl"
    val_in = base / "val.jsonl"
    test_in = base / "test.jsonl"
    meta_in = base / "meta.json"

    if not (train_in.exists() and val_in.exists() and test_in.exists() and meta_in.exists()):
        raise FileNotFoundError(f"Missing files in {base}")

    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]

    for r in ratios:
        tag = f"{int(r*100):03d}pct"
        out_dir = out_root / tag / "data" / "sliding_lstm"
        out_dir.mkdir(parents=True, exist_ok=True)

        # copy fixed files
        shutil.copy2(val_in, out_dir / "val.jsonl")
        shutil.copy2(test_in, out_dir / "test.jsonl")
        shutil.copy2(meta_in, out_dir / "meta.json")

        n, k = sample_jsonl(train_in, out_dir / "train.jsonl", r, seed=args.seed)
        print(f"[{tag}] train: {k}/{n} lines saved -> {out_dir/'train.jsonl'}")

if __name__ == "__main__":
    main()
