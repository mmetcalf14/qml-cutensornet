from typing import NamedTuple
from pytket.circuit import Circuit, OpType
import math

from quimb.tensor.circuit import CircuitMPS
from quimb.tensor import MatrixProductState, SpinHam1D
import quimb as qt
import numpy as np
import time

class Config(NamedTuple):
    value_of_zero: float

Z = qt.spin_operator('Z')
X = qt.spin_operator('X')
Y = qt.spin_operator('Y')
I = qt.spin_operator('I')

def build_ham_mpo_operator(num_qubits):

    ham_mpo_mat = []
    for i in range(num_qubits):
        hamx = SpinHam1D(S=1 / 2)
        hamx[i] += 1, 'X'
        mpo_hamx = hamx.build_mpo(num_qubits)
        
        hamy = SpinHam1D(S=1 / 2)
        hamy[i] += 1, 'Y'
        mpo_hamy = hamy.build_mpo(num_qubits)
        
        hamz = SpinHam1D(S=1 / 2)
        hamz[i] += 1, 'Z'
        mpo_hamz = hamz.build_mpo(num_qubits)
        
        ham_mpo_mat.append([mpo_hamx, mpo_hamy, mpo_hamz])
        
    print('mpo mat shape: ',len(ham_mpo_mat),len(ham_mpo_mat[0]))
        
    return ham_mpo_mat
    
    
def simulate(circ: Circuit, config: Config) -> MatrixProductState:
    """Simulate the circuit via MPS using Quimb on numpy backend.
    """

    # NOTE: Gates with parameters need to have math.pi multiplied to them, due to TKET's angle convention

    mps = CircuitMPS(circ.n_qubits)
    for gate in circ.get_commands():
        if gate.op.type == OpType.H:
            mps.h(gate.qubits[0].index[0])
        elif gate.op.type == OpType.Rx:
            mps.rx(math.pi*gate.op.params[0], gate.qubits[0].index[0])
        elif gate.op.type == OpType.Rz:
            mps.rz(math.pi*gate.op.params[0], gate.qubits[0].index[0])
        elif gate.op.type == OpType.ZZPhase:
            mps.rzz(math.pi*gate.op.params[0], gate.qubits[0].index[0], gate.qubits[1].index[0])
        elif gate.op.type == OpType.XXPhase:
            mps.rxx(math.pi*gate.op.params[0], gate.qubits[0].index[0], gate.qubits[1].index[0])
        elif gate.op.type == OpType.SWAP:
            mps.swap(gate.qubits[0].index[0], gate.qubits[1].index[0])
        else:
            raise Exception(f"Unknown gate {gate.op.type}")

    mps.psi.compress(cutoff=config.value_of_zero, cutoff_mode="abs")
    return mps.psi

def operator_expectation(mps_i, mps_j, qubit, nq, mpo_ham):

#    ham = SpinHam1D(S=1 / 2)
#    ham[qubit] += 1, operator
#    mpo_ham = ham.build_mpo(nq)
    
    psiH = mps_i.H
    mps_j.align_(mpo_ham, psiH)
    
    expectation_value = (psiH & mpo_ham & mps_j) ^ ...
    
    return expectation_value

def compute_1_rdm(circ: Circuit, config: Config, n_qubits, ham_mpos):
    rdms = np.zeros((n_qubits,2,2),dtype=complex)
    mps = simulate(circ, config)
    for i in range(n_qubits):
        start = time.time()
        expx = operator_expectation(mps, mps, qubit = i, nq=n_qubits, mpo_ham=ham_mpos[i][0])
        if i == 0 :
            print('exp time: ', time.time()-start)
#        expz = operator_expectation(mps, mps, qubit = i, nq=n_qubits, mpo_ham=ham_mpos[i][1])
#        expy = operator_expectation(mps, mps, qubit = i, nq=n_qubits, mpo_ham=ham_mpos[i][2])
#        rho = 0.5*(I + expx*X + expy*Y + expz*Z)
        rho = 0.5*(I + expx*X )
        rdms[i,:,:] = rho
    
    return rdms


