import matplotlib.pyplot as plt
import numpy as np

# Данные с маленьким и большим стандартным отклонением
data_low = np.random.normal(50, 5, 1000)   # Маленькое отклонение (5)
data_high = np.random.normal(50, 20, 1000)  # Большое отклонение (20)

plt.hist(data_low, bins=30, alpha=0.5, label="std=5", color="blue")
plt.hist(data_high, bins=30, alpha=0.5, label="std=20", color="red")

plt.legend()
plt.title("Гистограмма распределений с разным стандартным отклонением")
plt.show()