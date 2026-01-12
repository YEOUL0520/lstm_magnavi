import matplotlib.pyplot as plt
import numpy as np

thresholds = [1, 2, 3, 5]
lstm_cdf = [3.7, 8.8, 13.5, 22.6]
hyena_cdf = [41.4, 67.7, 77.4, 85.2]

plt.figure(figsize=(7,5))
plt.plot(thresholds, lstm_cdf, marker="o", label="LSTM")
plt.plot(thresholds, hyena_cdf, marker="o", label="Hyena")
plt.xlabel("Error threshold (m)")
plt.ylabel("CDF (%)")
plt.title("CDF Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
