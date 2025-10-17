import pandas as pd
import matplotlib.pyplot as plt

bt = pd.read_csv("bt_BL_vs_baselines_test.csv", parse_dates=["date"])
bt.set_index("date", inplace=True)

plt.figure(figsize=(10,6))
plt.plot(bt["cum_BL"], label="Black–Litterman (GRU)")
plt.plot(bt["cum_EW"], label="Equal Weight")
plt.plot(bt["cum_Sent"], label="Sentiment-only")

plt.title("Évolution du portefeuille (janv–oct 2025)")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
