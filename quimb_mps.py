from typing import NamedTuple
from pytket.circuit import Circuit, OpType

from quimb.tensor.circuit import CircuitMPS

class Config(NamedTuple):
    chi: int
    value_of_zero: float


def simulate(circ: Circuit, config: Config) -> CircuitMPS:
    """Simulate the circuit via MPS using Quimb on numpy backend.
    """
    mps = CircuitMPS(circ.n_qubits)
    for gate in circ.get_commands():
        if gate.op.type == OpType.H:
            mps.h(gate.qubits[0].index[0])
        elif gate.op.type == OpType.Rx:
            mps.rx(gate.op.params[0], gate.qubits[0].index[0])
        elif gate.op.type == OpType.Rz:
            mps.rz(gate.op.params[0], gate.qubits[0].index[0])
        elif gate.op.type == OpType.ZZPhase:
            mps.rzz(gate.op.params[0], gate.qubits[0].index[0], gate.qubits[1].index[0])
        elif gate.op.type == OpType.XXPhase:
            mps.rxx(gate.op.params[0], gate.qubits[0].index[0], gate.qubits[1].index[0])
        else:
            raise Exception(f"Unknown gate {gate.op.type}")

    mps.psi.compress(max_bond=config.chi, cutoff=config.value_of_zero, cutoff_mode="abs")
    return mps


