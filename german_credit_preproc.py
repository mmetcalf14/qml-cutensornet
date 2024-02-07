import numpy as np
import pandas as pd

feature_labels = ["acct_status", "duration","cred_hist", "purpose", "cred_amount", "savings", "curr_empl", "install_rate","marital_status","other_debt","curr_res","property", "age", "other_install", "housing","num_credit","job", "num_depend", "telephone","foreign", "Class"]
df = pd.read_csv("datasets/german_credit/german.data",sep=" ",names=feature_labels)

df.loc[df["Class"]==2, "Class"] = 0

df.loc[df["acct_status"] == "A11", "acct_status"] = 0
df.loc[df["acct_status"] == "A12", "acct_status"] = 100
df.loc[df["acct_status"] == "A13", "acct_status"] = 200
df.loc[df["acct_status"] == "A14", "acct_status"] = -1

df.loc[df["cred_hist"] == "A30", "cred_hist"] = 10
df.loc[df["cred_hist"] == "A31", "cred_hist"] = 8
df.loc[df["cred_hist"] == "A32", "cred_hist"] = 5
df.loc[df["cred_hist"] == "A33", "cred_hist"] = 1
df.loc[df["cred_hist"] == "A34", "cred_hist"] = 0

df.loc[df["savings"] == "A61", "savings"] = 10
df.loc[df["savings"] == "A62", "savings"] = 100
df.loc[df["savings"] == "A63", "savings"] = 500
df.loc[df["savings"] == "A64", "savings"] = 1000
df.loc[df["savings"] == "A65", "savings"] = -1

df.loc[df["curr_empl"] == "A71", "curr_empl"] = -1
df.loc[df["curr_empl"] == "A72", "curr_empl"] = 0
df.loc[df["curr_empl"] == "A73", "curr_empl"] = 1
df.loc[df["curr_empl"] == "A74", "curr_empl"] = 2
df.loc[df["curr_empl"] == "A75", "curr_empl"] = 3

#Note that guarantor and age correlate
#This might correlate to other debt features, maybe a combination with another feature here
df.loc[df["other_install"] == "A141", "other_install"] = 1
df.loc[df["other_install"] == "A142", "other_install"] = -1
df.loc[df["other_install"] == "A143", "other_install"] = 0

df.loc[df["housing"] == "A151", "housing"] = 5
df.loc[df["housing"] == "A152", "housing"] = 10
df.loc[df["housing"] == "A153", "housing"] = 0

df.loc[df["job"] == "A171", "job"] = -5
df.loc[df["job"] == "A172", "job"] = -1
df.loc[df["job"] == "A173", "job"] = 1
df.loc[df["job"] == "A174", "job"] = 5

df['agg_feat_1'] = df['acct_status']+df['savings']
df['agg_feat_2'] = df['curr_empl']+df['job']
df['agg_feat_3'] = df['cred_hist']+df['other_install']
df['agg_feat_4'] = df['acct_status']+df['duration']
df['agg_feat_5'] = df['cred_amount']+df['cred_hist']
df['agg_feat_6'] = df['curr_res']+df['housing']



df_one_hot = pd.get_dummies(df,columns=['property','purpose','marital_status', 'other_debt','foreign','telephone'],dtype='int')

df_one_hot.to_csv("datasets/german_credit_preproc.csv")
