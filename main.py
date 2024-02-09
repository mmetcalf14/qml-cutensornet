import numpy as np
import pandas as pd
import networkx as nx
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

def entanglement_graph(graph_type, nq, nn=None, ep=None, seed=None):
    """
    function to produce the edgelist/entanglement map for a circuit ansatz
    graph type (str): Either a random graph or a linear entanglement map
    nq (int): Number of qubits/features
    nn (int): number of nearest neighbors for linear entanglement map
    ep (float): Edge probability for random graph
    seed (int): Seed for the random graph
    """
    map = []

    if nn == None:
        nn = 1
    if ep == None:
        ep = 0.5

    if graph_type == 'random':
        graph = nx.gnp_random_graph(nq, ep, seed=seed)
        map = nx.edges(graph)
    elif graph_type == 'linear':
        map = [(i, i + j) for i in range(nq - 1) for j in range(1, nn + 1) if (i + j) < nq]
    else:
        raise RuntimeError("You have not specified a valid entanglement map")

    return map
    
def draw_sample(df, ndmin, ndmaj, test_frac=0.2, seed=123):
    """
    Function to sample from data and then divide into train/test sets
    df: Pandas dataframe
    ndmin (int): data size for minority class
    ndmaj (int): data size for majority class
    test_frac: fraction to divide data into train and test
    seed: random seed for sampling
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
edge_probability = float(sys.argv[4])
nearest_neighbors = int(sys.argv[5])
map_strategy = str(sys.argv[6])
entanglement_map = entanglement_graph(map_strategy,num_features,nn=nearest_neighbors, ep=edge_probability,seed=1235)

n_illicit = int(sys.argv[7])
n_licit = int(sys.argv[8])

data_seed = int(sys.argv[9])
data_file = str(sys.argv[10])


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

# Create the ansatz class
ansatz = KernelStateAnsatz(
	num_qubits=num_features,
	reps=reps,
	gamma=gamma,
	entanglement_map=entanglement_map,
	hadamard_init=True
)

train_info = "train_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit)
test_info = "test_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit)

time0 = MPI.Wtime()
kernel_train = build_kernel_matrix(config, ansatz, X = reduced_train_features, info_file=train_info, minutes_per_checkpoint=minutes_per_checkpoint, mpi_comm=mpi_comm)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on training set. Time: {round(time1-time0,2)} seconds\n")
    np.save("kernels/TrainKernel_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit),kernel_train)

time0 = MPI.Wtime()
kernel_test = build_kernel_matrix(config, ansatz, X = reduced_train_features, Y = reduced_test_features, info_file=test_info, minutes_per_checkpoint=minutes_per_checkpoint, mpi_comm=mpi_comm)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on test set. Time: {round(time1-time0,2)} seconds\n")
    np.save("kernels/TestKernel_Nf-{}_r-{}_g-{}_Ntr-{}.npy".format(num_features, reps, gamma, n_illicit),kernel_test)
    print('Test Kernel\n',kernel_test)

#############################
# Testing the kernel matrix #
#############################

if rank == root:
    reg = [4,3.5,3,2.5,2,1.5,1,0.5,0.1,0.05,0.01]
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

    np.save('data/TrainData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit),train_results)
    np.save('data/TestData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit),test_results)
