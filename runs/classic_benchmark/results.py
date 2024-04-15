from pathlib import Path
import numpy as np
from collections import defaultdict

directory_path = Path("raw/")

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

print("Results using Gaussian kernel:")
print(f"\tAUC: {max(avg_auc)}")
print(f"\trecall: {max(avg_recall)}")
print(f"\tprecision: {max(avg_precision)}")
print(f"\taccuracy: {max(avg_accuracy)}")

