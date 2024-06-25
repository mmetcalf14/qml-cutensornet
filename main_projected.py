import numpy as np
import pandas as pd
import sys
import pathlib
from mpi4py import MPI

import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
#from kernel_state_ansatz import KernelStateAnsatz, build_kernel_matrix
from projected_kernel_dense import ProjectedKernelStateAnsatz, build_kernel_matrix
import scipy.linalg as la

from quimb_mps import Config

import warnings
warnings.filterwarnings("ignore")

mpi_comm = MPI.COMM_WORLD
rank = mpi_comm.Get_rank()
n_procs = mpi_comm.Get_size()
root = 0

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

# Choose how many minutes separate different checkpoints
minutes_per_checkpoint = 30

# Simulation parameters.
config = Config(
    value_of_zero = 1e-16
)

# QML model parameters
num_features = int(sys.argv[1])
reps = int(sys.argv[2])
gamma = float(sys.argv[3])
alpha = float(sys.argv[4])
entanglement_map = [[i,i+1] for i in range(num_features-1)]

n_illicit = int(sys.argv[5])
n_licit = int(sys.argv[6])
data_seed = int(sys.argv[7])
data_file = str(sys.argv[8])

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
    print(f"\tn_illicit: {n_illicit}")
    print(f"\tn_licit: {n_licit}")
    print("")
    sys.stdout.flush()

#########################
# Load data and prepare #
#########################

# TODO: Should this be done only by process 0 and then broadcasted?
#  Not for now, this is not a bottleneck.
data = pd.read_csv('datasets/'+ data_file)

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

#################################
# Construction of kernel matrix #
#################################

pathlib.Path("kernels").mkdir(exist_ok=True)
pathlib.Path("data").mkdir(exist_ok=True)

# Create the ansatz class
ansatz = ProjectedKernelStateAnsatz(
	num_qubits=num_features,
    num_features=num_features,
	reps=reps,
	gamma=gamma,
	entanglement_map=entanglement_map,
	hadamard_init=True
)

train_info = "train_Nf-{}_r-{}_g-{}_Ntr-{}".format(num_features, reps, gamma, n_illicit)
test_info = "test_Nf-{}_r-{}_g-{}_Ntr-{}".format(num_features, reps, gamma, n_illicit)

time0 = MPI.Wtime()
kernel_train = build_kernel_matrix(mpi_comm, config, ansatz, X = reduced_train_features, alpha=alpha, info_file=train_info, minutes_per_checkpoint=minutes_per_checkpoint)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on training set. Time: {round(time1-time0,2)} seconds\n")
    np.save("kernels/TrainKernel_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit),kernel_train)
    #print(kernel_train)
    sys.stdout.flush()

time0 = MPI.Wtime()
kernel_test = build_kernel_matrix(mpi_comm, config, ansatz, X = reduced_train_features, Y = reduced_test_features, alpha=alpha, info_file=test_info, minutes_per_checkpoint=minutes_per_checkpoint)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on test set. Time: {round(time1-time0,2)} seconds\n")
    np.save("kernels/TestKernel_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit),kernel_test)
    #print('Test Kernel\n',kernel_test)
    sys.stdout.flush()

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

#    np.save('data/TrainData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit_train),train_results)
#    np.save('data/TestData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit_test),test_results)
