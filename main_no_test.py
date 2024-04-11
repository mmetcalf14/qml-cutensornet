from mpi4py import MPI

import numpy as np
import pandas as pd
import networkx as nx
import sys
import pathlib
import logging

import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import scipy.linalg as la

mpi_comm = MPI.COMM_WORLD
rank, n_procs = mpi_comm.Get_rank(), mpi_comm.Get_size()
root = 0

def entanglement_graph(nq, nn):
    """
    Function to produce the edgelist/entanglement map for a circuit ansatz

    Args:
        nq (int): Number of qubits/features.
        nn (int): Number of nearest neighbors for linear entanglement map.

    Returns:
        A list of pairs of qubits that should have a Rxx acting between them.
    """
    map = []
    for d in range(1, nn+1):  # For all distances from 1 to nn
        busy = set()  # Collect the right qubits of pairs on the first layer for this distance
        # Apply each gate between qubit i and its i+d (if it fits). Do so in two layers.
        for i in range(nq):
            if i not in busy and i+d < nq:  # All of these gates can be applied in one layer
                map.append((i, i+d))
                busy.add(i+d)
        # Apply the other half of the gates on distance d; those whose left qubit is in `busy`
        for i in busy:
            if i+d < nq:
                map.append((i, i+d))

    return map

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

# The truncation error assigned to the simulation
truncation_error = 1e-16

input_error_msg = (
    "\nCall script as \'python main.py <backend> <num_features> <layers> <gamma> <distance> <n_illicit> <n_licit> <data_seed> <data_file>\'."
    "\nThe value of <backend> must be either GPU or CPU."
)
if len(sys.argv) <= 9:
    raise ValueError(input_error_msg)

# QML model parameters
backend = str(sys.argv[1])
num_features = int(sys.argv[2])
reps = int(sys.argv[3])
gamma = float(sys.argv[4])
nearest_neighbors = int(sys.argv[5])
entanglement_map = entanglement_graph(nq=num_features, nn=nearest_neighbors)

n_illicit = int(sys.argv[6])
n_licit = int(sys.argv[7])
data_seed = int(sys.argv[8])
data_file = str(sys.argv[9])

if rank == root:
    print("\nUsing the following parameters:")
    print("")
    print(f"\tn_procs: {n_procs}")
    print(f"\tbackend: {backend}")
    print("")
    print(f"\tnum_features: {num_features}")
    print(f"\treps: {reps}")
    print(f"\tgamma: {gamma}")
    print(f"\tinteraction distance: {nearest_neighbors}")
    print(f"\tentanglement_map: {entanglement_map}")
    print("")
    print(f"\tn_illicit: {n_illicit}")
    print(f"\tn_licit: {n_licit}")
    print("")
    print(f"\tdata_seed: {data_seed}")
    print(f"\tdata_file: {data_file}")
    print("")
    sys.stdout.flush()

#########################
# Load data and prepare #
#########################

if backend == "GPU":
    from gpu_backend.kernel_state_ansatz import KernelStateAnsatz, build_kernel_matrix
elif backend == "CPU":
    from cpu_backend.kernel_state_ansatz import KernelStateAnsatz, build_kernel_matrix
else:
    raise ValueError(input_error_msg)

data = pd.read_csv('datasets/'+ data_file)

x_train, y_train, x_test, y_test = draw_sample(data,n_illicit, n_licit, 0.2, data_seed)

transformer = QuantileTransformer(output_distribution='normal')
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

minmax_scale = MinMaxScaler((0,2)).fit(x_train)
x_train = minmax_scale.transform(x_train)
x_test = minmax_scale.transform(x_test)

reduced_train_features = x_train[:,0:num_features]
reduced_test_features = x_test[:,0:num_features]

#################################
# Construction of kernel matrix #
#################################

pathlib.Path("kernels").mkdir(exist_ok=True)
pathlib.Path("data").mkdir(exist_ok=True)

# Create the ansatz class
ansatz = KernelStateAnsatz(
    num_qubits=num_features,
    reps=reps,
    gamma=gamma,
    entanglement_map=entanglement_map,
    hadamard_init=True
)

train_info = f"train_Nf{num_features}_r{reps}_g{gamma}_p0.0_nn{nearest_neighbors}_mslinear_Ntr{n_illicit}_s{data_seed}_{data_file.split('.')[0]}"
test_info = f"test_Nf{num_features}_r{reps}_g{gamma}_p0.0_nn{nearest_neighbors}_mslinear_Ntr{n_illicit}_s{data_seed}_{data_file.split('.')[0]}"

time0 = MPI.Wtime()
kernel_train = build_kernel_matrix(
    mpi_comm,
    ansatz,
    X=reduced_train_features,
    info_file=train_info,
    truncation_error=truncation_error
)
time1 = MPI.Wtime()
if rank == root:
    print(f"Built kernel matrix on training set. Time: {round(time1-time0,2)} seconds\n")
    np.save(f"kernels/{train_info}.npy", kernel_train)
