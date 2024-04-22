from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd

df_dict = {"depth": [], "AUC": [], "recall": [], "precision": [], "accuracy": []}

# Collect the results from the quantum kernels
directory_path = Path("raw/")

accuracy_dict = defaultdict(list)
precision_dict = defaultdict(list)
recall_dict = defaultdict(list)
auc_dict = defaultdict(list)

for file_path in directory_path.iterdir():

    filename = str(file_path).split(".npy")[0].split("/")[-1]

    flags = filename.split("_")
    if flags[0] != "test": continue  # Only evaluate metrics for test

    depth = int(flags[2][1:])

    results = np.load(file_path)
    for row in results:
        reg = row[0]
        accuracy_dict[(reg,depth)].append(row[1])
        precision_dict[(reg,depth)].append(row[2])
        recall_dict[(reg,depth)].append(row[3])
        auc_dict[(reg,depth)].append(row[4])

# For each possible depth value, obtain a row for the table
for depth in {2, 4, 8, 12, 16, 20}:

    # Obtain the average for each regularization coefficient
    avg_accuracy = [np.mean(metrics) for (_,x), metrics in accuracy_dict.items() if x==depth]
    avg_precision = [np.mean(metrics) for (_,x), metrics in precision_dict.items() if x==depth]
    avg_recall = [np.mean(metrics) for (_,x), metrics in recall_dict.items() if x==depth]
    avg_auc = [np.mean(metrics) for (_,x), metrics in auc_dict.items() if x==depth]

    # Among these, choose the one with highest AUC
    best_result_idx = avg_auc.index(max(avg_auc))

    # Add them to the dataframe dictionary
    df_dict["depth"].append(depth)
    df_dict["AUC"].append(avg_auc[best_result_idx])
    df_dict["recall"].append(avg_recall[best_result_idx])
    df_dict["precision"].append(avg_precision[best_result_idx])
    df_dict["accuracy"].append(avg_accuracy[best_result_idx])


# Create the DataFrame and dump to CSV file
df = pd.DataFrame.from_dict(df_dict)
print(df)
df.to_csv("results.csv", index=False)
