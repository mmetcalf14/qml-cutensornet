import pandas as pd
import matplotlib.pyplot as plt

df_gpu = pd.read_csv("gpu_results.csv")
df_gpu = df_gpu.sort_values(by="neighbours")
df_gpu = df_gpu.drop(columns=['features', 'entanglement', 'layers', 'gamma', 'seed'])
df_cpu = pd.read_csv("cpu_results.csv")
df_cpu = df_cpu.sort_values(by="neighbours")
df_cpu = df_cpu.drop(columns=['features', 'entanglement', 'layers', 'gamma', 'seed'])

# Chi comparison table
merged_df = pd.merge(df_gpu, df_cpu, left_on='neighbours', right_on='neighbours', how='outer', suffixes=("_gpu","_cpu"))
merged_df = merged_df.drop(columns=["median_dot_time_gpu", "q1_dot_time_gpu", "q3_dot_time_gpu", "median_mps_time_gpu", "q1_mps_time_gpu", "q3_mps_time_gpu", "median_dot_time_cpu", "q1_dot_time_cpu", "q3_dot_time_cpu", "median_mps_time_cpu", "q1_mps_time_cpu", "q3_mps_time_cpu"])
print(merged_df)

# Crossover for MPS sim
x = df_gpu["neighbours"]
y = df_gpu["median_mps_time"]
plt.plot(x, y, marker="o", color="mediumblue", label="GPU")
for x, y, y_q1, y_q3 in zip(x, y, df_gpu["q1_mps_time"], df_gpu["q3_mps_time"]):
  plt.errorbar(x, y, yerr=[[y - y_q1], [y_q3 - y]], color="mediumblue", alpha=0.35, capsize=5)

x = df_cpu["neighbours"]
y = df_cpu["median_mps_time"]
plt.plot(x, y, marker="o", color="orange", label="CPU")
for x, y, y_q1, y_q3 in zip(x, y, df_cpu["q1_mps_time"], df_cpu["q3_mps_time"]):
  plt.errorbar(x, y, yerr=[[y - y_q1], [y_q3 - y]], color="orange", alpha=0.35, capsize=5)

plt.xlabel("Interaction distance", fontsize=11)
plt.ylabel("MPS sim. time (s)", fontsize=11)
plt.yscale("log")
plt.legend(fontsize=10, loc="upper left")
plt.show()

# Crossover for inner products
x = df_gpu["neighbours"]
y = df_gpu["median_dot_time"]
plt.plot(x, y, marker="o", color="mediumblue", label="GPU")
for x, y, y_q1, y_q3 in zip(x, y, df_gpu["q1_dot_time"], df_gpu["q3_dot_time"]):
  plt.errorbar(x, y, yerr=[[y - y_q1], [y_q3 - y]], color="mediumblue", alpha=0.35, capsize=5)

x = df_cpu["neighbours"]
y = df_cpu["median_dot_time"]
plt.plot(x, y, marker="o", color="orange", label="CPU")
for x, y, y_q1, y_q3 in zip(x, y, df_cpu["q1_dot_time"], df_cpu["q3_dot_time"]):
  plt.errorbar(x, y, yerr=[[y - y_q1], [y_q3 - y]], color="orange", alpha=0.35, capsize=5)

plt.xlabel("Interaction distance", fontsize=11)
plt.ylabel("Inner product time (s)", fontsize=11)
plt.yscale("log")
plt.legend(fontsize=10, loc="upper left")
plt.show()
