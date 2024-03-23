import pandas as pd
import matplotlib.pyplot as plt

df_gpu = pd.read_csv("gpu.csv")
df_gpu = df_gpu.loc[df_gpu["layers"]==2].loc[df_gpu["features"]==50].loc[df_gpu["gamma"]==0.5].sort_values(by="neighbours")
df_cpu = pd.read_csv("cpu.csv")
df_cpu = df_cpu.loc[df_cpu["reps"]==2].loc[df_cpu["G"]==0.5].sort_values(by="nn")

# Chi comparison table
merged_df = pd.merge(df_gpu, df_cpu, left_on='neighbours', right_on='nn', how='inner', suffixes=("_gpu","_cpu"))
chi_df = merged_df.drop(columns=['features', 'entanglement', 'layers', 'gamma', 'seed', 'median_mps_time', 'median_dot_time', 'avg_mps_mem', 'reps', 'nn', 'G', 'Ansatz', 'mps sim time', 'dot prod time'])
chi_df["proportion"] = chi_df["avg_max_chi_gpu"] / chi_df["avg_max_chi_cpu"]
print(chi_df)

# Crossover for MPS sim
plt.plot(df_gpu["neighbours"], df_gpu["median_mps_time"], marker="o", label="GPU")
plt.plot(df_cpu["nn"], df_cpu["mps sim time"], marker="o", label="CPU")

plt.xlabel("Interaction distance")
plt.ylabel("MPS sim. time (s)")
plt.yscale("log")
plt.legend()
plt.show()

# Crossover for inner products
plt.plot(df_gpu["neighbours"], df_gpu["median_dot_time"], marker="o", label="GPU")
plt.plot(df_cpu["nn"], df_cpu["dot prod time"], marker="o", label="CPU")

plt.xlabel("Interaction distance")
plt.ylabel("Inner product time (s)")
plt.yscale("log")
plt.legend()
plt.show()

# Chi comparison plot
#width=0.35
#df_gpu = df_gpu.loc[df_gpu["neighbours"]<=8]
#df_cpu = df_cpu.loc[df_cpu["nn"]<=8]

#plt.bar(df_gpu["neighbours"]-width, df_gpu["avg_max_chi"], 2*width, label="GPU")
#plt.bar(df_cpu["nn"]+width, df_cpu["avg_max_chi"], 2*width, label="CPU")

#plt.xlabel("Interaction distance")
#plt.xticks([2,4,6,8])
#plt.ylabel("Max. virtual bond dim.")
#plt.yscale("log")
#plt.legend()
#plt.show()