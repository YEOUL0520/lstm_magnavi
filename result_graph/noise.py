import matplotlib.pyplot as plt
import numpy as np

noise = [0, 1, 5, 10, 20]
lstm_mae = [13.334, 13.332, 13.342, 13.331, 13.413]
hyena_mae = [3.983, 3.973, 3.993, 4.144, 4.689]

plt.figure(figsize=(7,5))
plt.plot(noise, lstm_mae, marker="o", label="LSTM")
plt.plot(noise, hyena_mae, marker="o", label="Hyena")
plt.xlabel("Noise level (%)")
plt.ylabel("MAE (m)")
plt.title("Noise Robustness")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
