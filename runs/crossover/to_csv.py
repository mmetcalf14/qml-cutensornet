import os
import json
import pandas as pd
import numpy as np

# Gather the data from the GPU runs
data_dict = {
  "features": [], "entanglement": [],
  "layers": [], "gamma": [], "neighbours": [], "seed": [],
  "median_mps_time": [], "median_dot_time": [], "avg_max_chi": [], "avg_mps_mem": [],
  "q1_mps_time": [], "q3_mps_time": [], "q1_dot_time": [], "q3_dot_time": [],
}

kernel = "train"
dir = "raw/gpu/"
for f in os.listdir(dir):
    filename = os.fsdecode(f).split(".json")[0]

    flags = filename.split("_")
    data_dict["features"].append(int(flags[1][2:]))
    data_dict["layers"].append(int(flags[2][1:]))
    data_dict["gamma"].append(float(flags[3][1:]))
    data_dict["neighbours"].append(int(flags[5][2:]))
    data_dict["entanglement"].append(flags[6][2:])
    data_dict["seed"].append(int(flags[8][1:]))

    with open(dir+filename+".json") as f:
      this_run = json.load(f)

    data_dict["median_mps_time"].append(this_run['median_circ_sim'][0])
    data_dict["q1_mps_time"].append(this_run['q1_circ_sim'][0])
    data_dict["q3_mps_time"].append(this_run['q3_circ_sim'][0])
    data_dict["median_dot_time"].append(this_run['median_product'][0])
    data_dict["q1_dot_time"].append(this_run['q1_product'][0])
    data_dict["q3_dot_time"].append(this_run['q3_product'][0])
    data_dict["avg_max_chi"].append((this_run['ave max chi x'][0] + this_run['ave max chi y'][0]) / 2)
    data_dict["avg_mps_mem"].append(this_run['avg_mps_mem'][0])

df = pd.DataFrame.from_dict(data_dict)

print(df)
df.to_csv("gpu_results.csv", index=False)


# Gather the data from the CPU runs
data_dict = {
  "features": [], "entanglement": [],
  "layers": [], "gamma": [], "neighbours": [], "seed": [],
  "median_mps_time": [], "median_dot_time": [], "avg_max_chi": [],
  "q1_mps_time": [], "q3_mps_time": [], "q1_dot_time": [], "q3_dot_time": [],
}

kernel = "train"
dir = "raw/cpu/"
for f in os.listdir(dir):
    filename = os.fsdecode(f).split(".json")[0]

    flags = filename.split("_")
    data_dict["features"].append(int(flags[1][2:]))
    data_dict["layers"].append(int(flags[2][1:]))
    data_dict["gamma"].append(float(flags[3][1:]))
    data_dict["neighbours"].append(int(flags[5][2:]))
    data_dict["entanglement"].append(flags[6][2:])
    data_dict["seed"].append(int(flags[8][1:]))

    with open(dir+filename+".json") as f:
      this_run = json.load(f)

    data_dict["median_mps_time"].append(this_run['median_circ_sim'][0])
    data_dict["q1_mps_time"].append(this_run['q1_circ_sim'][0])
    data_dict["q3_mps_time"].append(this_run['q3_circ_sim'][0])
    data_dict["median_dot_time"].append(this_run['median_product'][0])
    data_dict["q1_dot_time"].append(this_run['q1_product'][0])
    data_dict["q3_dot_time"].append(this_run['q3_product'][0])
    data_dict["avg_max_chi"].append((this_run['ave max chi x'][0] + this_run['ave max chi y'][0]) / 2)

df = pd.DataFrame.from_dict(data_dict)

print(df)
df.to_csv("cpu_results.csv", index=False)
