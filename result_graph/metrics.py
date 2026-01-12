import matplotlib.pyplot as plt
import numpy as np

metrics = ["MAE", "RMSE", "P50", "P90"]
lstm = [13.334, 16.954, 10.718, 27.040]
hyena = [3.983, 9.340, 1.253, 8.949]

x = np.arange(len(metrics))
w = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - w/2, lstm, w, label="LSTM")
plt.bar(x + w/2, hyena, w, label="Hyena")
plt.xticks(x, metrics)
plt.ylabel("Error (m)")
plt.title("LSTM vs Hyena (EUC Distance)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
