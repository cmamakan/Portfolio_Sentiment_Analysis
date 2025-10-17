import pandas as pd
import matplotlib.pyplot as plt

bt_lstm = pd.read_csv("bt_LSTM.csv")
bt_gru  = pd.read_csv("bt_GRU.csv")
bt_rnn  = pd.read_csv("bt_RNN.csv")

plt.figure(figsize=(8,5))
plt.plot(bt_lstm["date"], bt_lstm["cum_BL_LSTM"], label="LSTM")
plt.plot(bt_gru["date"], bt_gru["cum_BL_GRU"], label="GRU")
plt.plot(bt_rnn["date"], bt_rnn["cum_BL_RNN"], label="RNN")
plt.title("Évolution du portefeuille Black–Litterman (2025)")
plt.xlabel("Date"); plt.ylabel("Valeur cumulée")
plt.legend(); plt.grid(True)
plt.show()