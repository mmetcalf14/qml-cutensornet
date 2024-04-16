import numpy as np
import pandas as pd
import sys
import pathlib

import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score


def draw_sample(df, ndmin, ndmaj, test_frac=0.2, seed=123):
    """
    Function to sample from data and then divide into train/test sets.

    Args:
        df: Pandas dataframe
        ndmin (int): data size for minority class
        ndmaj (int): data size for majority class
        test_frac: fraction to divide data into train and test
        seed: random seed for sampling

    Returns:
        List of samples
    """
    data_reduced = pd.concat([df[df['Class']==0].sample(ndmin ,random_state=(seed*20+2)), df[df['Class']==1].sample(ndmaj,  random_state=(seed*46+9))], axis=0)
    train_df, test_df = train_test_split(data_reduced,  stratify=data_reduced['Class'], test_size=test_frac ,random_state=seed*26+19)
    train_labels = train_df.pop('Class')
    test_labels = test_df.pop('Class')
    
    return np.array(train_df), np.array(train_labels,dtype='int'), np.array(test_df), np.array(test_labels,dtype='int')
    
##############
# Parameters #
##############

# QML model parameters
num_features = int(sys.argv[1])

n_illicit = int(sys.argv[2])
n_licit = int(sys.argv[3])

data_seed = int(sys.argv[4])
data_file = sys.argv[5]


print("\nUsing the following parameters:")
print("")
print(f"\tnum_features: {num_features}")
print("")
print(f"\tn_illicit: {n_illicit}")
print(f"\tn_licit: {n_licit}")
print("")
print(f"\tdata_seed: {data_seed}")
print("")

#########################
# Load data and prepare #
#########################

data = pd.read_csv('../../datasets/'+ data_file)
#data.pop("Unnamed: 0")

train_features, train_labels, test_features, test_labels = draw_sample(data,n_illicit, n_licit, 0.2, data_seed)

transformer = QuantileTransformer(output_distribution='normal')
train_features = transformer.fit_transform(train_features)
test_features = transformer.transform(test_features)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

minmax_scale = MinMaxScaler((0,2)).fit(train_features)
train_features = minmax_scale.transform(train_features)
test_features = minmax_scale.transform(test_features)

reduced_train_features = train_features[:,0:num_features]
reduced_test_features = test_features[:,0:num_features]


#############################
# Testing the kernel matrix #
#############################

reg = [4,3.5,3,2.5,2,1.5,1,0.5,0.1,0.05,0.01]
test_results = []
for key, r in enumerate(reg):
    svc = SVC(kernel="rbf", C=r, tol=1e-3, gamma="scale", verbose=False)
    # In rbf, if gamma='scale', then it uses 1 / (n_features * X.var()) as value of gamma

    svc.fit(reduced_train_features, train_labels)
    test_predict = svc.predict(reduced_test_features)
    accuracy = accuracy_score(test_labels,test_predict)
    precision = precision_score(test_labels,test_predict)
    recall = recall_score(test_labels, test_predict)
    auc = roc_auc_score(test_labels, test_predict)
    test_results.append([r,accuracy, precision, recall, auc])

np.save(f"raw/gaussian/seed_{data_seed}.npy", test_results)
