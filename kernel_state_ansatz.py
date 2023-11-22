import sys
import json
import pathlib
import time as t

import numpy as np
from sympy import Symbol

from pytket import Circuit
from pytket.circuit import PauliExpBox, Pauli

from quimb_mps import Config, simulate

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
                exponent = (1/np.pi)*gamma*self.feature_symbol_list[i]
                self.ansatz_circ.Rz(exponent, i)

            for (q0, q1) in entanglement_map:
                symb0 = self.feature_symbol_list[q0]
                symb1 = self.feature_symbol_list[q1]
                exponent = gamma*gamma*(1 - symb0)*(1 - symb1)
                self.ansatz_circ.XXPhase(exponent, q0, q1)

    def circuit_for_data(self, feature_values: list[float]) -> Circuit:
        """Produce the circuit with its symbols being replaced by the given values.
        """
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError("The number of values must match the number of symbols.")

        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)

        return the_circuit


def build_kernel_matrix(
        config: Config,
        ansatz: KernelStateAnsatz,
        X,
        Y=None,
        info_file=None,
        minutes_per_checkpoint=None,
    ) -> np.ndarray:
    """Calculation of entries of the kernel matrix.

    Notes:
        For matters of efficiency, it is assumed that `len(Y) <= len(X)`. If this is not
        the case, consider swapping them and applying a conjugate transposition to the output.

        By default, if `Y` is not provided, it is set to `X`. Using this default option when
        possible is preferable for matters of efficiency.

    Args:
        config: An instance of ConfigMPS setting the configuration of simulations.
        ansatz: a symbolic circuit describing the ansatz.
        X: A 2D array where `X[i, :]` corresponds to the i-th data point and
            each `X[:, j]` corresponds to the values of the j-th feature across
            all data points.
        Y: A 2D array where `Y[i, :]` corresponds to the i-th data point and
            each `Y[:, j]` corresponds to the values of the j-th feature across
            all data points. If not provided it is set to be equal to `X`.
        info_file: The name of the file where to save performance information of this call.
            If not provided, the performance information will only appear in stdout.
        minutes_per_checkpoint: The amount of time (in minutes) elapsed between different
            checkpoint saves of the kernel matrix. If None, no checkpoints are saved.

    Returns:
        A kernel matrix of dimensions `len(X)`x`len(Y)`.
    """

    rank = 0
    n_procs = 1
    root = 0
    device_id = 0

    # Checkpointing file
    pathlib.Path("tmp").mkdir(exist_ok=True)
    checkpoint_file = pathlib.Path(f"tmp/checkpoint_rank_{rank}_" + info_file)

    # Dictionary to keep track of profiling information
    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (len(X), "entries")
        profiling_dict["lenY"] = (None if Y is None else len(Y), "entries")
        start_time = t.time()

    # We use different parallelisation strategies depending on whether Y set to X or not

    ###########################################
    # Calculate the kernel matrix when X == Y #
    ###########################################
    if Y is None:
        time0 = t.time()

        n_circs = len(X)
        circs_per_proc = int(np.ceil(n_circs / n_procs))
        this_proc_circs = []

        # Create each of the circuits to be contracted by this GPUs
        for i in range(rank*circs_per_proc, (rank+1)*circs_per_proc):
            if i < n_circs:
                this_proc_circs.append(ansatz.circuit_for_data(X[i, :]))
            else:
                this_proc_circs.append(None)

        if rank == root:
            time1 = t.time()
            print(f"[Rank 0] Circuit list generated. Time taken: {round(time1-time0,2)} seconds.")
            profiling_dict["r0_circ_gen"] = (time1-time0, "seconds")
            print("\nContracting the MPS of the circuits...")
            sys.stdout.flush()
            time0 = t.time()

        # Contract the MPS of each of the circuits in this process
        this_proc_mps = []
        progress_bar = 0
        progress_checkpoint = int(np.ceil(len(this_proc_circs) / 10))

        for k, circ in enumerate(this_proc_circs):
            # Simulate the circuit and obtain the output state as an MPS
            if circ is not None:
                mps = simulate(circ, config)
            else:
                mps = None
            this_proc_mps.append(mps)

            if rank == root and progress_bar * progress_checkpoint < k:
                print(f"{progress_bar*10}%")
                sys.stdout.flush()
                progress_bar += 1

        if rank == root:
            time1 = t.time()
            print("100%")
            duration = time1-time0
            print(f"[Rank 0] MPS contracted. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_circ_sim"] = (duration, "seconds")
            average = duration / circs_per_proc
            print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
            profiling_dict["avg_circ_sim"] = (average, "seconds")

            #mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in this_proc_mps]
            #total_bytes = sum(mps_byte_size) / (1024**2)
            #print(f"[Rank 0] Total memory currently used: {round(total_bytes,2)} MiB")
            #avg_bytes_per_mps = total_bytes / len(mps_byte_size)
            #print(f"\tAverage MPS memory footprint: {round(avg_bytes_per_mps,2)} MiB")
            #profiling_dict["avg_mps_mem"] = (avg_bytes_per_mps, "MiB")

            print("\nBroadcasting the MPS of the circuits.")
            sys.stdout.flush()
            time0 = t.time()

        mps_list = this_proc_mps

        if rank == root:
            time1 = t.time()
            print(f"[Rank 0] MPS broadcasted in {round(time1-time0,2)} seconds")
            profiling_dict["r0_broadcast"] = (time1-time0, "seconds")

            #mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in mps_list]
            #total_bytes = sum(mps_byte_size) / (1024**2)
            #print(f"Total MPS memory used per GPU: {round(total_bytes,2)} MiB")
            #profiling_dict["gpu_mps_mem"] = (total_bytes, "MiB")

        # Enumerate all pairs of circuits to be overlapped
        pairs = [(i, j) for i in range(n_circs) for j in range(n_circs) if i < j]

        # Try to recover from the last checkpoint (if any)
        if checkpoint_file.is_file():
            # Load the kernel matrix from the checkpoint file (one per process)
            kernel_mat = np.load(checkpoint_file)
            print(f"[Rank {rank}] Recovered from checkpoint!")
        else:
            # Allocate space for kernel matrix
            kernel_mat = np.zeros(shape=(n_circs, n_circs))
        last_checkpoint_time = t.time()

        if rank == root:
            print("\nObtaining inner products...")
            time0 = t.time()

        # Parallelise across all available processes
        pairs_per_proc = int(np.ceil(len(pairs) / n_procs))
        progress_bar, progress_checkpoint = 0, int(np.ceil(pairs_per_proc / 10))
        for k in range(rank*pairs_per_proc, (rank+1)*pairs_per_proc):

            if k >= len(pairs): break

            # Skip if this value was saved in the checkpoint
            (i, j) = pairs[k]
            if kernel_mat[i, j] != 0: continue

            #if k% 1000 == 0: print(f"Iteration {k} yields data pair {pairs[k]} on process {rank}")
            # Run contraction
            mps0 = mps_list[i]
            mps1 = mps_list[j]

            timea = t.time()
            overlap = mps0.psi.H @ mps1.psi
            kernel_mat[i, j] = kernel_mat[j, i] = (overlap*np.conj(overlap)).real
            timeb = t.time()
            #print(f"[{t.time()}] Took {timeb - timea} seconds on process {rank}, device {device_id}.")

            # Save a checkpoint if it's time
            if minutes_per_checkpoint is not None and last_checkpoint_time + 60*minutes_per_checkpoint < t.time():
                last_checkpoint_time = t.time()

                # Remove the previous checkpoint file
                checkpoint_file.unlink(missing_ok=True)
                # Create a new checkpoint
                np.save(checkpoint_file, kernel_mat)
                # Inform user
                print(f"[Rank {rank}] Checkpoint saved at {checkpoint_file}!")

            # Report back to user
            if rank == root and progress_bar * progress_checkpoint < k:
                print(f"{progress_bar*10}%")
                sys.stdout.flush()
                progress_bar += 1

        if rank == root:
            time1 = t.time()
            print("100%")
            duration = time1 - time0
            print(f"[Rank 0] Inner products calculated. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_product"] = (duration, "seconds")
            average = duration / pairs_per_proc
            print(f"\tAverage time per inner product: {round(average,4)} seconds.\n")
            profiling_dict["avg_product"] = (average, "seconds")
            np.fill_diagonal(kernel_mat,1)


    ###########################################
    # Calculate the kernel matrix when X != Y #
    ###########################################
    else:
        time0 = t.time()

        x_circs = len(X)
        y_circs = len(Y)
        x_circs_per_proc = int(np.ceil(x_circs / n_procs))
        y_circs_per_proc = int(np.ceil(y_circs / n_procs))
        this_proc_x_circs = []
        this_proc_y_circs = []

        # Create each of the circuits from X to contract in this GPU
        for i in range(rank*x_circs_per_proc, (rank+1)*x_circs_per_proc):
            if i < x_circs:
                this_proc_x_circs.append(ansatz.circuit_for_data(X[i, :]))
            else:
                this_proc_x_circs.append(None)

        # Create each of the circuits from Y to contract in this GPU
        for i in range(rank*y_circs_per_proc, (rank+1)*y_circs_per_proc):
            if i < y_circs:
                this_proc_y_circs.append(ansatz.circuit_for_data(Y[i, :]))
            else:
                this_proc_y_circs.append(None)

        if rank == root:
            time1 = t.time()
            print(f"[Rank 0] Circuit list generated. Time taken: {time1-time0} seconds.")
            profiling_dict["r0_circ_gen"] = (time1-time0, "seconds")
            print("\nContracting the MPS of the circuits...")
            sys.stdout.flush()
            time0 = t.time()

        # Contract the MPS of each of the circuits in this process
        this_proc_x_mps = []
        this_proc_y_mps = []
        progress_bar = 0
        this_proc_circs = this_proc_x_circs + this_proc_y_circs
        progress_checkpoint = int(np.ceil(len(this_proc_circs) / 10))

        for k, circ in enumerate(this_proc_circs):
            # Simulate the circuit and obtain the output state as an MPS
            if circ is not None:
                mps = simulate(circ, config)
            else:
                mps = None

            if k < len(this_proc_x_circs):
                this_proc_x_mps.append(mps)
            else:
                this_proc_y_mps.append(mps)

            if rank == root and progress_bar * progress_checkpoint < k:
                print(f"{progress_bar*10}%")
                sys.stdout.flush()
                progress_bar += 1

        if rank == root:
            time1 = t.time()
            print("100%")
            duration = time1-time0
            print(f"[Rank 0] MPS contracted. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_circ_sim"] = (duration, "seconds")
            average = duration / (x_circs_per_proc + y_circs_per_proc)
            print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
            profiling_dict["avg_circ_sim"] = (average, "seconds")

            #mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in this_proc_x_mps + this_proc_y_mps]
            #total_bytes = sum(mps_byte_size) / (1024**2)
            #print(f"[Rank 0] Total memory currently used: {round(total_bytes,2)} MiB")
            #avg_bytes_per_mps = total_bytes / len(mps_byte_size)
            #print(f"\tAverage MPS memory footprint: {round(avg_bytes_per_mps,2)} MiB")
            #profiling_dict["avg_mps_mem"] = (avg_bytes_per_mps, "MiB")

            print("\nBroadcasting the MPS of the circuits.")
            sys.stdout.flush()
            time0 = t.time()

        # The MPS from the X dataset need not be broadcasted, those from Y do.
        # Remove any `None` previously introduced for padding
        this_proc_x_mps = [mps for mps in this_proc_x_mps if mps is not None]

        y_mps_list = this_proc_y_mps

        if rank == root:
            time1 = t.time()
            print(f"[Rank 0] MPS broadcasted in {round(time1-time0,2)} seconds")
            profiling_dict["r0_broadcast"] = (time1-time0, "seconds")

            #mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in this_proc_x_mps + y_mps_list]
            #total_bytes = sum(mps_byte_size) / (1024**2)
            #print(f"Total MPS memory used per GPU: {round(total_bytes,2)} MiB")
            #profiling_dict["gpu_mps_mem"] = (total_bytes, "MiB")

        # Try to recover from the last checkpoint (if any)
        if checkpoint_file.is_file():
            # Load the kernel matrix from the checkpoint (one per process)
            kernel_mat = np.load(checkpoint_file)
            print(f"[Rank {rank}] Recovered from checkpoint!")
        else:
            # Allocate space for kernel matrix
            kernel_mat = np.zeros(shape=(y_circs, x_circs))
        last_checkpoint_time = t.time()

        if rank == root:
            print("\nObtaining inner products...")
            time0 = t.time()

        # Each process will calculate the inner products of its MPS from X with
        # all of the MPS from Y.
        progress_bar = 0
        progress_checkpoint = int(np.ceil(len(this_proc_x_mps) / 10))

        for i, x_mps in enumerate(this_proc_x_mps):
            for j, y_mps in enumerate(y_mps_list):

                # Skip if this value was saved in the checkpoint
                if kernel_mat[j, i + rank*x_circs_per_proc] != 0: continue

                timea = t.time()
                overlap = x_mps.psi.H @ y_mps.psi
                kernel_mat[j, i + rank*x_circs_per_proc] = (overlap*np.conj(overlap)).real
                timeb = t.time()
                #print(f"[{t.time()}] Took {timeb - timea} seconds on process {rank}, device {device_id}.")

                # Save a checkpoint if it's time
                if minutes_per_checkpoint is not None and last_checkpoint_time + 60*minutes_per_checkpoint < t.time():
                    last_checkpoint_time = t.time()

                    # Remove the previous checkpoint file
                    checkpoint_file.unlink(missing_ok=True)
                    # Create a new checkpoint
                    np.save(checkpoint_file, kernel_mat)
                    # Inform user
                    print(f"[Rank {rank}] Checkpoint saved at {checkpoint_file}!")

                # Report back to user
                if rank == root and progress_bar * progress_checkpoint < i:
                    print(f"{progress_bar*10}%")
                    sys.stdout.flush()
                    progress_bar += 1

        if rank == root:
            time1 = t.time()
            print("100%")
            duration = time1 - time0
            print(f"[Rank 0] Inner products calculated. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_product"] = (duration, "seconds")
            average = duration / (len(this_proc_x_mps) * len(y_mps_list))
            print(f"\tAverage time per inner product: {round(average,4)} seconds.\n")
            profiling_dict["avg_product"] = (average, "seconds")

    if rank == root:
        end_time = t.time()
        profiling_dict["total_time"] = (end_time-start_time, "seconds")

    # If requested by user, dump `profiling_dict` to file
    if info_file is not None and rank == root:
        with open(info_file, 'w') as fp:
            json.dump(profiling_dict, fp, indent=4)

    # We can delete the checkpoint file (useful, so that we avoid risk of collisions)
    checkpoint_file.unlink(missing_ok=True)

    return kernel_mat
