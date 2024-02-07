import pandas as pd
import numpy as np

data = pd.read_csv('datasets/creditcard.csv')
data.pop("Time")
epsilon = 0.001
data["Amount"] = np.log(data.pop("Amount")+epsilon)

for key,val in enumerate(data):
    if key !=0 and val != 'Class' and val != 'Amount':
        data[f"A_{key}"] = data[temp_val]+data[val]
    temp_val = val

data.to_csv("datasets/cct_agg.csv")

