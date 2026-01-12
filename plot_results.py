import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_ratio(tag: str) -> float:
    """
    tag: '100pct', '050pct', '020pct', '010pct' 등
    return: 100.0, 50.0, 20.0, 10.0
    """
    if tag is None:
        return float("nan")
    m = re.search(r"(\d+)\s*pct", str(tag))
    if not m:
        return float("nan")
    return float(int(m.group(1)))


def main():
    # ====== 경로 설정 ======
    csv_path = Path("exp_results/scarcity/_eval/results.csv")
    out_dir = Path("exp_results/scarcity/_eval/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    # ====== 로드 ======
    df = pd.read_csv(csv_path)

    # 필수 컬럼 체크
    required = ["arch", "tag", "euc_mae", "euc_cdf_5m"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in results.csv: {missing_cols}")

    # ratio 숫자화
    df["ratio"] = df["tag"].apply(parse_ratio)

    # 정리: 필요한 컬럼만 + 최신(중복 실행 대비)
    # (tag+arch 조합이 여러 번 있을 수 있으니, 마지막 행을 사용)
    df = df.sort_values(by=["arch", "ratio"]).dropna(subset=["ratio"])
    df_last = df.groupby(["arch", "ratio"], as_index=False).tail(1)

    # LSTM / HYENA 분리
    # 저장할 때 arch가 'lstm'/'hyena'로 들어갔다고 가정
    df_last["arch"] = df_last["arch"].str.lower()
    lstm = df_last[df_last["arch"] == "lstm"].copy()
    hyena = df_last[df_last["arch"] == "hyena"].copy()

    # ratio 정렬
    lstm = lstm.sort_values("ratio")
    hyena = hyena.sort_values("ratio")

    # ====== Plot 1: EUC MAE vs ratio ======
    plt.figure()
    if len(lstm) > 0:
        plt.plot(lstm["ratio"], lstm["euc_mae"], marker="o", label="LSTM")
    if len(hyena) > 0:
        plt.plot(hyena["ratio"], hyena["euc_mae"], marker="o", label="Hyena")

    plt.xlabel("Train data ratio (%)")
    plt.ylabel("EUC MAE (m)")
    plt.title("Data Scarcity: EUC MAE vs Train Ratio")
    plt.xticks(sorted(df_last["ratio"].unique()))
    plt.grid(True)
    plt.legend()
    mae_path = out_dir / "euc_mae_vs_ratio.png"
    plt.savefig(mae_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ====== Plot 2: EUC CDF<=5m vs ratio ======
    plt.figure()
    if len(lstm) > 0:
        plt.plot(lstm["ratio"], lstm["euc_cdf_5m"], marker="o", label="LSTM")
    if len(hyena) > 0:
        plt.plot(hyena["ratio"], hyena["euc_cdf_5m"], marker="o", label="Hyena")

    plt.xlabel("Train data ratio (%)")
    plt.ylabel("EUC CDF ≤ 5m (%)")
    plt.title("Data Scarcity: CDF ≤ 5m vs Train Ratio")
    plt.xticks(sorted(df_last["ratio"].unique()))
    plt.grid(True)
    plt.legend()
    cdf_path = out_dir / "euc_cdf5_vs_ratio.png"
    plt.savefig(cdf_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ====== Plot 3: MAN 그래프 ======
    if "man_mae" in df_last.columns and "man_cdf_5m" in df_last.columns:
         plt.figure()
         if len(lstm) > 0:
             plt.plot(lstm["ratio"], lstm["man_mae"], marker="o", label="LSTM")
         if len(hyena) > 0:
             plt.plot(hyena["ratio"], hyena["man_mae"], marker="o", label="Hyena")
         plt.xlabel("Train data ratio (%)")
         plt.ylabel("MAN MAE (m)")
         plt.title("Data Scarcity: MAN MAE vs Train Ratio")
         plt.xticks(sorted(df_last["ratio"].unique()))
         plt.grid(True)
         plt.legend()
         plt.savefig(out_dir / "man_mae_vs_ratio.png", dpi=200, bbox_inches="tight")
         plt.close()
    
         plt.figure()
         if len(lstm) > 0:
             plt.plot(lstm["ratio"], lstm["man_cdf_5m"], marker="o", label="LSTM")
         if len(hyena) > 0:
             plt.plot(hyena["ratio"], hyena["man_cdf_5m"], marker="o", label="Hyena")
         plt.xlabel("Train data ratio (%)")
         plt.ylabel("MAN CDF ≤ 5m (%)")
         plt.title("Data Scarcity: MAN CDF ≤ 5m vs Train Ratio")
         plt.xticks(sorted(df_last["ratio"].unique()))
         plt.grid(True)
         plt.legend()
         plt.savefig(out_dir / "man_cdf5_vs_ratio.png", dpi=200, bbox_inches="tight")
         plt.close()

    print("✅ Saved plots:")
    print(f" - {mae_path}")
    print(f" - {cdf_path}")
    print(f"📁 Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
