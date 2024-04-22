from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd

df_dict = {"kernel": [], "d": [], "gamma": [], "AUC": [], "recall": [], "precision": [], "accuracy": []}

# Collect the results from the Gaussian kernel
directory_path = Path("raw/gaussian")

accuracy_dict = defaultdict(list)
precision_dict = defaultdict(list)
recall_dict = defaultdict(list)
auc_dict = defaultdict(list)

for file_path in directory_path.iterdir():
    results = np.load(file_path)
    for row in results:
        reg = row[0]
        accuracy_dict[reg].append(row[1])
        precision_dict[reg].append(row[2])
        recall_dict[reg].append(row[3])
        auc_dict[reg].append(row[4])

# Obtain the average for each regularization coefficient
avg_accuracy = [np.mean(metrics) for metrics in accuracy_dict.values()]
avg_precision = [np.mean(metrics) for metrics in precision_dict.values()]
avg_recall = [np.mean(metrics) for metrics in recall_dict.values()]
avg_auc = [np.mean(metrics) for metrics in auc_dict.values()]

# Among these, choose the one with highest AUC
best_result_idx = avg_auc.index(max(avg_auc))

# Add them to the dataframe dictionary
df_dict["kernel"].append("Gaussian")
df_dict["gamma"].append("--")
df_dict["d"].append("--")
df_dict["AUC"].append(avg_auc[best_result_idx])
df_dict["recall"].append(avg_recall[best_result_idx])
df_dict["precision"].append(avg_precision[best_result_idx])
df_dict["accuracy"].append(avg_accuracy[best_result_idx])



# Collect the results from the quantum kernels
directory_path = Path("raw/quantum")

accuracy_dict = defaultdict(list)
precision_dict = defaultdict(list)
recall_dict = defaultdict(list)
auc_dict = defaultdict(list)

for file_path in directory_path.iterdir():

    filename = str(file_path).split(".npy")[0].split("/")[-1]

    flags = filename.split("_")
    if flags[0] != "test": continue  # Only evaluate metrics for test

    gamma = float(flags[3][1:])
    d = int(flags[5][2:])

    results = np.load(file_path)
    for row in results:
        reg = row[0]
        accuracy_dict[(reg,gamma,d)].append(row[1])
        precision_dict[(reg,gamma,d)].append(row[2])
        recall_dict[(reg,gamma,d)].append(row[3])
        auc_dict[(reg,gamma,d)].append(row[4])

# For each possible d and gamma, obtain a row for the table
for gamma in {0.1,0.5,1.0}:
  for d in {1,2,4,6}:

    # Obtain the average for each regularization coefficient
    avg_accuracy = [np.mean(metrics) for (_,x,y), metrics in accuracy_dict.items() if x==gamma and y==d]
    avg_precision = [np.mean(metrics) for (_,x,y), metrics in precision_dict.items() if x==gamma and y==d]
    avg_recall = [np.mean(metrics) for (_,x,y), metrics in recall_dict.items() if x==gamma and y==d]
    avg_auc = [np.mean(metrics) for (_,x,y), metrics in auc_dict.items() if x==gamma and y==d]

    # Among these, choose the one with highest AUC
    best_result_idx = avg_auc.index(max(avg_auc))

    # Add them to the dataframe dictionary
    df_dict["kernel"].append("quantum")
    df_dict["d"].append(d)
    df_dict["gamma"].append(gamma)
    df_dict["AUC"].append(avg_auc[best_result_idx])
    df_dict["recall"].append(avg_recall[best_result_idx])
    df_dict["precision"].append(avg_precision[best_result_idx])
    df_dict["accuracy"].append(avg_accuracy[best_result_idx])


# Create the DataFrame and dump to CSV file
df = pd.DataFrame.from_dict(df_dict)
print(df)
df.to_csv("results.csv", index=False)
