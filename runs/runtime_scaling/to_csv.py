import os
import json
import pandas as pd
import numpy as np

data_dict = {
  "data_set": [], "kernel": [], "n_procs": [],
  "features": [], "licit_data": [], "entanglement": [],
  "layers": [], "gamma": [], "edge_prob": [], "neighbours": [], "seed": [],
  "mps_wall_time": [], "dot_wall_time": [], "mpi_wall_time": [], "total_wall_time": [],
  "avg_max_chi": [], "avg_mps_mem": [],
}

dir = "raw/"
kernel = "train"
for f in os.listdir(dir):
    filename = os.fsdecode(f).split(".json")[0]

    data_dict["kernel"].append(kernel)

    flags = filename.split("_")
    data_dict["features"].append(int(flags[1][2:]))
    data_dict["layers"].append(int(flags[2][1:]))
    data_dict["gamma"].append(float(flags[3][1:]))
    data_dict["edge_prob"].append(float(flags[4][1:]))
    data_dict["neighbours"].append(int(flags[5][2:]))
    data_dict["entanglement"].append(flags[6][2:])
    data_dict["licit_data"].append(int(flags[7][3:]))
    data_dict["seed"].append(int(flags[8][1:]))
    data_dict["data_set"].append(flags[9].split(".csv")[0])

    with open(dir+filename+".json") as f:
      this_run = json.load(f)

    data_dict["n_procs"].append(this_run['n_procs'][0])
    data_dict["mps_wall_time"].append(this_run['r0_circ_sim'][0])
    data_dict["dot_wall_time"].append(this_run['r0_product'][0])
    data_dict["mpi_wall_time"].append(this_run['r0_RR_recv'][0])
    data_dict["total_wall_time"].append(this_run['total_time'][0])
    data_dict["avg_mps_mem"].append(this_run['avg_mps_mem'][0])
    data_dict["avg_max_chi"].append((this_run['ave max chi x'][0] + this_run['ave max chi y'][0]) / 2)

df = pd.DataFrame.from_dict(data_dict)

print(df)
df.to_csv("results.csv", index=False)
