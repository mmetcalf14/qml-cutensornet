import numpy as np
import pandas as pd
from mpi4py import MPI
import sys

import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import scipy.linalg as la

from sympy import Symbol
from pytket import Circuit, OpType
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle, SimulationAlgorithm, Config, simulate


mpi_comm = MPI.COMM_WORLD
rank, n_procs = mpi_comm.Get_rank(), mpi_comm.Get_size()
root = 0


class KernelStateAnsatz:
    """Class that creates and stores a symbolic ansatz circuit and can be used to
    produce instances of the circuit U(x)|0> for given parameters.

    Attributes:
        ansatz_circ: The symbolic circuit to be used as ansatz.
        feature_symbol_list: The list of symbols in the circuit, each corresponding to a feature.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int,
        gamma: float,
        entanglement_map: list[tuple[int, int]],
        hadamard_init: bool=True,
    ):
        """Generate the ansatz circuit and store it. The circuit has as many symbols as qubits, which
        is also the same number of features in the data set. Multiple gates will use the same symbols;
        particularly, 1-qubit gates acting on qubit `i` all use the same symbol, and two qubit gates
        acting qubits `(i,j)` will use the symbols for feature `i` and feature `j`.

        Args:
            num_qubits: number of qubits is the number of features to be encoded.
            reps: number of times to repeat the layer of unitaries.
            gamma: hyper parameter in unitary to be optimized for overfitting.
            entanglement_map: pairs of qubits to be entangled together in the ansatz,
                for now limit entanglement only to two body terms
            hadamard_init: whether a layer of H gates should be applied to all qubits
                at the beginning of the circuit.
        """

        self.one_q_symbol_list = []
        self.two_q_symbol_list = []

        self.ansatz_circ = Circuit(num_qubits)
        self.feature_symbol_list = [Symbol('f_'+str(i)) for i in range(num_qubits)]

        if hadamard_init:
            for i in range(num_qubits):
                self.ansatz_circ.H(i)

        for _ in range(reps):
            for i in range(num_qubits):
                exponent = (2/np.pi)*gamma*self.feature_symbol_list[i]
                self.ansatz_circ.Rz(exponent, i)

            for (q0, q1) in entanglement_map:
                symb0 = self.feature_symbol_list[q0]
                symb1 = self.feature_symbol_list[q1]
                exponent = gamma*gamma*(1 - symb0)*(1 - symb1)
                self.ansatz_circ.XXPhase(exponent, q0, q1)

        # Apply routing by adding SWAPs eagerly just before the XXPhase gates
        qubit_pos = {q: p for p, q in enumerate(self.ansatz_circ.qubits)}
        routed_circ = Circuit(self.ansatz_circ.n_qubits)  # The new circuit

        for cmd in self.ansatz_circ.get_commands():
            # Add it directly to the circuit if it's not an Rxx (aka XXPhase) gate
            if cmd.op.type != OpType.XXPhase:
                routed_circ.add_gate(cmd.op, cmd.qubits)
            # If it is Rxx, add SWAPs as necessary
            else:
                q0 = qubit_pos[cmd.qubits[0]]
                q1 = qubit_pos[cmd.qubits[1]]
                (q0, q1) = (min(q0,q1), max(q0,q1))
                # Add SWAP gates
                for q in range(q0, q1-1):
                    routed_circ.SWAP(q,q+1)
                # Apply XXPhase gate
                routed_circ.add_gate(cmd.op, [q1-1,q1])
                # Apply SWAP gates on the opposite order to return qubit to position
                for q in reversed(range(q0, q1-1)):
                    routed_circ.SWAP(q,q+1)

                self.ansatz_circ = routed_circ


    def circuit_for_data(self, feature_values: list[float]) -> Circuit:
        """Produce the circuit with its symbols being replaced by the given values.
        """
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError("The number of values must match the number of symbols.")

        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)

        return the_circuit


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

config = Config(
  loglevel=10,  # Activate debug mode to collect mem data
  value_of_zero=0,
  truncation_fidelity=1-truncation_error
)

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
circ_index = int(sys.argv[10])

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
    print(f"\tcirc_index: {circ_index}")
    print("")
    sys.stdout.flush()

#########################
# Load data and prepare #
#########################

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

#######################
# Running the circuit #
#######################

# Create the ansatz class
ansatz = KernelStateAnsatz(
    num_qubits=num_features,
    reps=reps,
    gamma=gamma,
    entanglement_map=entanglement_map,
    hadamard_init=True
)

# Simulate the circuit
circ = ansatz.circuit_for_data(reduced_train_features[circ_index,:])
with CuTensorNetHandle() as libhandle:
  mps = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, config)
