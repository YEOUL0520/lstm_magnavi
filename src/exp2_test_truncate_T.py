#!/usr/bin/env python3
"""
Exp2) Sequence length truncation test (NO re-preprocess)
- Evaluate existing checkpoint while truncating input to last T steps.
- Run for both LSTM/Hyena with same checkpoints.
"""

import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["lstm", "hyena"])
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data-dir", required=True, type=str)
    ap.add_argument("--Ts", type=str, default="250,150,75,30")
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-noise-test", action="store_true")
    args = ap.parse_args()

    Ts = [int(x.strip()) for x in args.Ts.split(",") if x.strip()]

    for T in Ts:
        print("\n" + "=" * 90)
        print(f"[Exp2] arch={args.arch} | T={T}")
        print("=" * 90)

        # test_only_unified.py에 --truncate-last-t 옵션을 추가해서 쓰는 방식
        cmd = [
            "python", "src/test_only_unified.py",
            "--arch", args.arch,
            "--checkpoint", args.checkpoint,
            "--data-dir", args.data_dir,
            "--batch-size", str(args.batch_size),
            "--truncate-last-t", str(T),
        ]
        if args.cpu:
            cmd.append("--cpu")
        if args.no_noise_test:
            cmd.append("--no-noise-test")

        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
