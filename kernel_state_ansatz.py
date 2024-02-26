from julia.api import Julia
jl = Julia(compiled_modules=False)  # TODO: Hopefully this can be avoided
from julia import KernelPkg

from typing import Optional
from mpi4py import MPI

import sys
import json
import pathlib
from statistics import median, mean

import numpy as np
from sympy import Symbol

from pytket import Circuit
from pytket.circuit import OpType
from pytket.transform import Transform
from pytket.architecture import Architecture
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit
from pytket.circuit import PauliExpBox, Pauli


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
                exponent = (2/np.pi)*gamma*gamma*(1 - symb0)*(1 - symb1)
                self.ansatz_circ.XXPhase(exponent, q0, q1)

        # Apply TKET routing to compile circuit to line architecture
        cu = CompilationUnit(self.ansatz_circ)
        architecture = Architecture(
            [(i, i + 1) for i in range(self.ansatz_circ.n_qubits - 1)]
        )
        DefaultMappingPass(architecture).apply(cu)
        self.ansatz_circ = cu.circuit
        Transform.DecomposeBRIDGE().apply(self.ansatz_circ)

    def circuit_for_data(
        self,
        feature_values: list[float]
    ) -> list[tuple[str, list[int], list[float]]]:
        """Produce the circuit with its symbols being replaced by the given values.

        Returns:
            The circuit as a list of gates that can be parsed by the KernelPkg. Each
            gate is represented by a tuple `(name, qubits, params)`.
        """
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError("The number of values must match the number of symbols.")

        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)

        gates = []
        for g in the_circuit.get_commands():
            qubits = [q.index[0] for q in g.qubits]
            if g.op.type == OpType.H:
                gates.append(("H", qubits, []))
            elif g.op.type == OpType.Rx:
                gates.append(("Rx", qubits, g.op.params))
            elif g.op.type == OpType.Rz:
                gates.append(("Rz", qubits, g.op.params))
            elif g.op.type == OpType.XXPhase:
                gates.append(("XXPhase", qubits, g.op.params))
            elif g.op.type == OpType.ZZPhase:
                gates.append(("ZZPhase", qubits, g.op.params))
            elif g.op.type == OpType.SWAP:
                gates.append(("SWAP", qubits, []))
            else:
                raise RuntimeError(f"Unrecognised {g.op.type}.")

        return gates


def build_kernel_matrix(
        mpi_comm,
        ansatz: KernelStateAnsatz,
        X,
        Y=None,
        info_file="info_file",
        value_of_zero: float=1e-16,
        number_of_tiles: Optional[int]=None,
    ) -> np.ndarray:
    """Calculation of entries of the kernel matrix.

    Notes:
        By default, if `Y` is not provided, it is set to `X`. Using this default option when
        possible is preferable for matters of efficiency.

    Args:
        mpi_comm: The MPI communicator.
        ansatz: A symbolic circuit describing the ansatz.
        X: A 2D array where `X[i, :]` corresponds to the i-th data point and
            each `X[:, j]` corresponds to the values of the j-th feature across
            all data points.
        Y: A 2D array where `Y[i, :]` corresponds to the i-th data point and
            each `Y[:, j]` corresponds to the values of the j-th feature across
            all data points. If not provided it is set to be equal to `X`.
        info_file: The name of the file where to save performance information of this call.
            Also used as a suffix for the checkpointing file. Defaults to "info_file".
        value_of_zero: The absolute cutoff below which singular values are removed.
        number_of_tiles: Determines a lower bound of the number of tiles the kernel matrix
            is split into. This should often be a multiple of the number of processes, so
            that each process is assigned the same number of tiles. Larger tiles (i.e. less
            tiles) tend to lead to better performance, as long as there is enough tiles for
            each process. Checkpoints are saved between computation of tiles. If a value is
            not provided, it defaults to 4x the number of processes.
    Returns:
        A kernel matrix of dimensions `len(Y)`x`len(X)`.
    """

    # MPI variables
    n_procs = mpi_comm.Get_size()
    rank = mpi_comm.Get_rank()
    root = 0

    # Distribution strategy parameters.
    lenX = len(X)
    lenY = lenX if Y is None else len(Y)
    number_of_tiles = number_of_tiles if number_of_tiles is not None else 4*n_procs
    # Distribution proceeds by tiling the kernel matrix into approximately
    # `number_of_tiles` square tiles and distributing the task of computing these
    # uniformly across the `n_procs` CPUs.
    tile_side = int(np.floor(
        # We want each tile to contain approx `|X|*|Y|/number_of_tiles` vdots.
        # The above leads to the "area" of each tile; its side is just the sqrt.
        np.sqrt(lenX*lenY / number_of_tiles)
    ))
    # Calculate the number of "columns" and "rows" that result from the tiling
    x_slices = int(np.ceil(lenX / tile_side))
    y_slices = int(np.ceil(lenY / tile_side))

    # Create the tile slices
    tiles = [  # Each tile is a tuple (y_slice, x_slice); a slice given by (start, end)
        (
            (y*tile_side, min(lenY, (y+1)*tile_side)),
            (x*tile_side, min(lenX, (x+1)*tile_side)),
        )
        for y in range(y_slices) for x in range(x_slices)
    ]
    n_tiles = len(tiles)
    # If X == Y, then thanks to symmetry we can ignore the tiles in the upper triangle
    if Y is None:  # Only keep tiles whose x_slice_start <= y_slice_start
        tiles = [t for t in tiles if t[1][0] <= t[0][0]]

    # Generate the list of circuits
    x_circs = [ansatz.circuit_for_data(X[i, :]) for i in range(len(X))]
    if Y is None:
        y_circs = x_circs
    else:
        y_circs = [ansatz.circuit_for_data(Y[i, :]) for i in range(len(Y))]

    # Checkpointing file
    pathlib.Path("tmp").mkdir(exist_ok=True)
    checkpoint_file = pathlib.Path(f"tmp/checkpoint_rank_{rank}_" + info_file + ".npy")

    # Dictionary to keep track of profiling information
    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (lenX, "entries")
        profiling_dict["lenY"] = (None if Y is None else lenY, "entries")
        profiling_dict["n_tiles"] = (n_tiles, "tiles")
        profiling_dict["value_of_zero"] = (value_of_zero, "")
        profiling_dict["vdots_per_tile"] = (tile_side**2, "entries")

        print(f"\nKernel matrix split into {n_tiles} tiles of {tile_side**2} entries each.")

    # Try to recover from the last checkpoint (if any)
    if checkpoint_file.is_file():
        # Load the kernel matrix from the checkpoint file (one per process)
        kernel_mat = np.load(checkpoint_file)
        print(f"[Rank {rank}] Recovered from checkpoint!")
    else:  # Otherwise, allocate space for kernel matrix
        kernel_mat = np.zeros(shape=(lenY, lenX))

    # Each process picks a tile and computes it
    tile_times = []
    start_time = MPI.Wtime()
    all_chi_x = []
    all_chi_y = []
    all_time_x = []
    all_time_y = []
    all_time_vdot = []
    for k, (y_slice, x_slice) in enumerate(tiles):
        if k % n_procs == rank:  # Otherwise, this process is not meant to compute this tile

            # Inform the user of progress
            if rank == root:
                print(f"{int(100*k/len(tiles))}%")
                sys.stdout.flush()

            # Check if the tile has already been computed in the checkpoint
            if kernel_mat[y_slice[0], x_slice[0]] != 0:
                continue  # If so, skip this iteration
            time0 = MPI.Wtime()

            # Otherwise, compute the tile
            tile_ix = np.ix_(range(*y_slice), range(*x_slice))
            kernel_mat[tile_ix], chi_x, chi_y, time_x, time_y, time_vdot = KernelPkg.compute_tile(
                ansatz.ansatz_circ.n_qubits,
                x_circs[x_slice[0]:x_slice[1]],
                y_circs[y_slice[0]:y_slice[1]],
                value_of_zero,
            )
            all_chi_x += chi_x
            all_chi_y += chi_y
            all_time_x += time_x
            all_time_y += time_y
            all_time_vdot += time_vdot

            # If the kernel matrix is symmetrix (X==Y) and this is not a diagonal tile
            if Y is None and x_slice[0] != y_slice[0]:
                # The tile on the other side of the diagonal is just the transpose
                tile_T_ix = np.ix_(range(*x_slice), range(*y_slice))
                kernel_mat[tile_T_ix] = kernel_mat[tile_ix].T

            # Record the time
            tile_times.append(MPI.Wtime() - time0)

            # Remove the previous checkpoint file
            checkpoint_file.unlink(missing_ok=True)
            # Create a new checkpoint
            np.save(checkpoint_file, kernel_mat)
            # Inform user
            #print(f"Checkpoint saved at {checkpoint_file}!")

    # Combine the kernel matrices of all processes
    kernel_mat = mpi_comm.reduce(kernel_mat, op=MPI.SUM, root=root)

    # Record time
    if rank == root:
        print("100%")
        end_time = MPI.Wtime()
        total_time = end_time - start_time
        profiling_dict["total_time"] = (total_time, "seconds")

        med_tile_time = median(tile_times)
        profiling_dict["median_tile_time"] = (med_tile_time, "seconds")
        print(f"[Rank {rank}] Median tile time is {round(med_tile_time,2)} seconds.")
        median_circ_sim = median(all_time_x + all_time_y)
        profiling_dict["median_circ_sim"] = (median_circ_sim, "seconds")
        median_product = median(all_time_vdot)
        profiling_dict["median_product"] = (median_product, "seconds")
        print(f"\tThe median MPS simulation time is {median_circ_sim} seconds.")
        print(f"\tThe median of vdot execution time is {median_product} seconds.")


        profiling_dict["ave max chi x"] = (mean(all_chi_x),"chi x")
        profiling_dict["ave max chi y"] = (mean(all_chi_y),"chi y")
        print(f"\tAverage max bond dimension x is {mean(all_chi_x)}")
        print(f"\tAverage max bond dimension y is {mean(all_chi_y)}")

    # Dump `profiling_dict` to file
    if rank == root:
        with open(info_file+".json", 'w') as fp:
            json.dump(profiling_dict, fp, indent=4)

    # We can delete the checkpoint file (so that we avoid risk of collisions)
    checkpoint_file.unlink(missing_ok=True)

    return kernel_mat
