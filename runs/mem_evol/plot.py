import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Interaction distance 6
directory = Path('raw/d6')
data_circuits_d6 = dict()
for filepath in directory.iterdir():
    if filepath.is_file():
        with open(filepath) as f:
            mps_sizes = []
            for l in f.readlines():
                if "MPS size" in l:
                    this_size = float(l.split("=")[1])
                    mps_sizes.append(this_size)
            data_circuits_d6[str(filepath)] = mps_sizes

nticks_on_completion = max(len(data) for data in data_circuits_d6.values())
progress_bar = [100*i/nticks_on_completion for i in range(nticks_on_completion)]

data_mean = [np.mean(data_tuple) for data_tuple in zip(*data_circuits_d6.values())]
plt.plot(progress_bar, data_mean, linewidth=0.75, color="mediumblue", label="d=6")
data_min = [np.min(data_tuple) for data_tuple in zip(*data_circuits_d6.values())]
data_max = [np.max(data_tuple) for data_tuple in zip(*data_circuits_d6.values())]
plt.fill_between(progress_bar, data_min, data_max, color='mediumblue', alpha=0.2)


# Interaction distance 12
directory = Path('raw/d12')
data_circuits_d12 = dict()
for filepath in directory.iterdir():
    if filepath.is_file():
        with open(filepath) as f:
            mps_sizes = []
            for l in f.readlines():
                if "MPS size" in l:
                    this_size = float(l.split("=")[1])
                    mps_sizes.append(this_size)
            data_circuits_d12[str(filepath)] = mps_sizes

nticks_on_completion = max(len(data) for data in data_circuits_d12.values())
progress_bar = [100*i/nticks_on_completion for i in range(nticks_on_completion)]

data_mean = [np.mean(data_tuple) for data_tuple in zip(*data_circuits_d12.values())]
plt.plot(progress_bar, data_mean, linewidth=0.75, color="orange", label="d=12")
data_min = [np.min(data_tuple) for data_tuple in zip(*data_circuits_d12.values())]
data_max = [np.max(data_tuple) for data_tuple in zip(*data_circuits_d12.values())]
plt.fill_between(progress_bar, data_min, data_max, color='orange', alpha=0.2)


plt.legend(fontsize=10)
plt.xlabel("Gates applied (%)", fontsize=11)
plt.ylabel("MPS size (MiB)", fontsize=11)
plt.yscale("log")
plt.show()

