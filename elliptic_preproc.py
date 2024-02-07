import pandas as pd
import numpy as np

feature_labels = []
feature_labels.append('Node')
feature_labels.append('Time')
for i in range(165):
    feature_labels.append('Feature {}'.format(i+1))

feature_data = pd.read_csv('datasets/elliptic_bitcoin_dataset/elliptic_txs_features.csv', names = feature_labels)
node_label = ['Node', 'Class']
node_class = pd.read_csv('datasets/elliptic_bitcoin_dataset/elliptic_txs_classes.csv', names = node_label)

node_class.loc[node_class["Class"] == "unknown", "Class"] = 99
node_class.loc[node_class["Class"] == "1", "Class"] = 0
node_class.loc[node_class["Class"] == "2", "Class"] = 1

clean_feature_data = feature_data.drop(np.where(node_class['Class']==99)[0])
clean_class_label = node_class.drop(np.where(node_class['Class']==99)[0])
del feature_data, node_class

elliptic_data = pd.merge(clean_class_label, clean_feature_data)
node = elliptic_data.pop('Node')
time = elliptic_data.pop('Time')

elliptic_data.to_csv('datasets/elliptic_preproc.csv')
