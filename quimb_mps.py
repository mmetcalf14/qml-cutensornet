from typing import NamedTuple
from pytket.circuit import Circuit, OpType
import math

from quimb.tensor.circuit import CircuitMPS
from quimb.tensor import MatrixProductState

class Config(NamedTuple):
    chi: int
    value_of_zero: float


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

    mps.psi.compress(max_bond=config.chi, cutoff=config.value_of_zero, cutoff_mode="abs")
    return mps.psi


