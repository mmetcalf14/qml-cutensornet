import numpy as np
import pandas as pd
from mpi4py import MPI
import sys
import logging

import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from kernel_state_ansatz import KernelStateAnsatz, build_kernel_matrix
import scipy.linalg as la

from pytket.extensions.cutensornet.mps import ConfigMPS

mpi_comm = MPI.COMM_WORLD
rank, n_procs = mpi_comm.Get_rank(), mpi_comm.Get_size()
root = 0

##############
# Parameters #
##############

# Choose how many minutes separate different checkpoints
minutes_per_checkpoint = 30

# Set up cuQuantum logger
logging.basicConfig(level=30)  # 30=quiet, 20=info, 10=debug

# Simulation parameters.
# See docs in: https://cqcl.github.io/pytket-cutensornet/api/modules/mps.html#pytket.extensions.cutensornet.mps.ConfigMPS
config = ConfigMPS(
    chi = 8,
    float_precision = np.float64,
#   value_of_zero = 1e-16  # 1e-16 is the default value for np.float64. Uncomment to change it.
    loglevel = 30,  # pytket-cutensornet logger. 30=quiet, 20=info, 10=debug
)

# QML model parameters
num_features = int(sys.argv[1])
reps = int(sys.argv[2])
gamma = float(sys.argv[3])
entanglement_map = [[i,i+1] for i in range(num_features-1)]

n_illicit_train = int(sys.argv[4])
n_licit_train = int(sys.argv[5])
n_illicit_test = int(sys.argv[6])
n_licit_test = int(sys.argv[7])

train_size = n_licit_train+n_illicit_train
test_size = n_licit_test+n_illicit_test
train_ratio = n_illicit_train/train_size
test_ratio = n_illicit_test/test_size

if rank == root:
    print("\nUsing the following parameters:")
    print("")
    print(f"\tn_procs: {n_procs}")
    print("")
    print(f"\tnum_features: {num_features}")
    print(f"\treps: {reps}")
    print(f"\tgamma: {gamma}")
    print(f"\tentanglement_map: {entanglement_map}")
    print("")
    print(f"\tn_illicit_train: {n_illicit_train}")
    print(f"\tn_licit_train: {n_licit_train}")
    print(f"\tn_illicit_test: {n_illicit_test}")
    print(f"\tn_licit_test: {n_licit_test}")
    print("")
    sys.stdout.flush()

#########################
# Load data and prepare #
#########################

# TODO: Should this be done only by process 0 and then broadcasted?
#  Not for now, this is not a bottleneck.

feature_labels = []
feature_labels.append('Node')
feature_labels.append('Time')
for i in range(165):
    feature_labels.append('Feature {}'.format(i+1))

feature_data = pd.read_csv('elliptic_bitcoin_dataset/elliptic_txs_features.csv', names = feature_labels)
node_label = ['Node', 'Class']
node_class = pd.read_csv('elliptic_bitcoin_dataset/elliptic_txs_classes.csv', names = node_label)

node_class.loc[node_class["Class"] == "unknown", "Class"] = 99
node_class.loc[node_class["Class"] == "1", "Class"] = 0
node_class.loc[node_class["Class"] == "2", "Class"] = 1

clean_feature_data = feature_data.drop(np.where(node_class['Class']==99)[0])
clean_class_label = node_class.drop(np.where(node_class['Class']==99)[0])
del feature_data, node_class

elliptic_data = pd.merge(clean_class_label, clean_feature_data)
node = elliptic_data.pop('Node')
time = elliptic_data.pop('Time')
del clean_feature_data, clean_class_label

data_reduced = pd.concat([elliptic_data[elliptic_data['Class']==0].sample(n_illicit_train+n_illicit_test,random_state=321), elliptic_data[elliptic_data['Class']==1].sample(n_licit_train+n_licit_test,  random_state=568)], axis=0)
train_df, test_df = train_test_split(data_reduced,  stratify=data_reduced['Class'], test_size=n_illicit_test+n_licit_test,random_state=345)

train_labels = np.array(train_df.pop('Class'),dtype='int')
test_labels = np.array(test_df.pop('Class'),dtype='int')
train_features = np.array(train_df)
test_features = np.array(test_df)

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

#################################
# Construction of kernel matrix #
#################################

# Create the ansatz class
ansatz = KernelStateAnsatz(
	num_qubits=num_features,
	reps=reps,
	gamma=gamma,
	entanglement_map=entanglement_map,
	hadamard_init=True
)

train_info = "train_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit_train)
test_info = "test_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit_test)

time0 = MPI.Wtime()
kernel_train = build_kernel_matrix(config, ansatz, X = reduced_train_features, info_file=train_info, minutes_per_checkpoint=minutes_per_checkpoint, mpi_comm=mpi_comm)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on training set. Time: {round(time1-time0,2)} seconds\n")
    np.save("kernels/TrainKernel_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit_train),kernel_train)

time0 = MPI.Wtime()
kernel_test = build_kernel_matrix(config, ansatz, X = reduced_train_features, Y = reduced_test_features, info_file=test_info, minutes_per_checkpoint=minutes_per_checkpoint, mpi_comm=mpi_comm)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on test set. Time: {round(time1-time0,2)} seconds\n")
    np.save("kernels/TestKernel_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit_test),kernel_test)
    print('Test Kernel\n',kernel_test)

#############################
# Testing the kernel matrix #
#############################

if rank == root:
    reg = [2,1.5,1,0.5,0.1,0.05,0.01]
    test_results = []
    for key, r in enumerate(reg):
        print('coeff: ', r)
        svc = SVC(kernel="precomputed", C=r, tol=1e-5, verbose=False)
        # scale might work best as 1/Nfeautres

        svc.fit(kernel_train, train_labels)
        test_predict = svc.predict(kernel_test)
        accuracy = accuracy_score(test_labels,test_predict)
        print('accuracy: ', accuracy)
        precision = precision_score(test_labels,test_predict)
        print('precision: ', precision)
        recall = recall_score(test_labels, test_predict)
        print('recall: ', recall)
        auc = roc_auc_score(test_labels, test_predict)
        print('auc: ', auc)
        test_results.append([r,accuracy, precision, recall, auc])

    train_results = []
    print('\n Train Results\n')
    for key, r in enumerate(reg):
        print('coeff: ', r)
        svc = SVC(kernel="precomputed", C=r, tol=1e-5, verbose=False)
        # scale might work best as 1/Nfeautres

        svc.fit(kernel_train, train_labels)
        test_predict = svc.predict(kernel_train)
        accuracy = accuracy_score(train_labels,test_predict)
        print('accuracy: ', accuracy)
        precision = precision_score(train_labels,test_predict)
        print('precision: ', precision)
        recall = recall_score(train_labels, test_predict)
        print('recall: ', recall)
        auc = roc_auc_score(train_labels, test_predict)
        print('auc: ', auc)
        train_results.append([r,accuracy, precision, recall, auc])

    np.save('data/TrainData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit_train),train_results)
    np.save('data/TestData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit_test),test_results)
