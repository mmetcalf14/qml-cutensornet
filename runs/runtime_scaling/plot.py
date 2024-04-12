import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Train kernel
df_train = df.loc[df["kernel"]=="train"].sort_values(by="n_procs")
plt.bar(["400 / 2", "800 / 4", "1600 / 8", "3200 / 16", "6400 / 32"], df_train["mpi_wall_time"]/3600, bottom=(df_train["mps_wall_time"]+df_train["dot_wall_time"])/3600, label="MPI communication", color="darkgreen", alpha=0.7)
plt.bar(["400 / 2", "800 / 4", "1600 / 8", "3200 / 16", "6400 / 32"], df_train["dot_wall_time"]/3600, bottom=df_train["mps_wall_time"]/3600, label="Inner products", color="orange", alpha=0.7)
plt.bar(["400 / 2", "800 / 4", "1600 / 8", "3200 / 16", "6400 / 32"], df_train["mps_wall_time"]/3600, label="MPS simulation", color="mediumblue", alpha=0.7)

plt.xlabel("Data size / num. GPUs", fontsize=11)
plt.ylabel("Runtime (hours)", fontsize=11)
plt.legend(fontsize=10)
plt.show()
