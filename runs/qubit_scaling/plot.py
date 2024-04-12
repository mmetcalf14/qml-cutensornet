import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df = df.loc[df["neighbours"]==6]

# Gamma = 0.1
df_g01 = df.loc[df["gamma"]==0.1].sort_values(by="features")
plt.plot(df_g01["features"], df_g01["avg_mps_time"], marker="o", color="mediumblue", label="0.1")

# Gamma = 0.5
df_g05 = df.loc[df["gamma"]==0.5].sort_values(by="features")
plt.plot(df_g05["features"], df_g05["avg_mps_time"], marker="o", color="orange", label="0.5")

# Gamma = 1.0
df_g1 = df.loc[df["gamma"]==1.0].sort_values(by="features")
plt.plot(df_g1["features"], df_g1["avg_mps_time"], marker="o", color="darkgreen", label="1.0")

plt.legend(title="$\gamma$", fontsize=10)
plt.xticks([30,60,90,120,150,165])
plt.xlabel("Number of qubits", fontsize=11)
plt.ylabel("MPS sim. time (s)", fontsize=11)
plt.show()