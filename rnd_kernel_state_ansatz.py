import sys
import json

from mpi4py import MPI
from cupy.cuda.runtime import getDeviceCount

import cupy as cp
import numpy as np
from sympy import Symbol

import cuquantum as cq
from pytket import Circuit
from pytket.circuit import PauliExpBox, Pauli
from pytket.extensions.cutensornet.mps import CuTensorNetHandle, ContractionAlg, simulate

class KernelStateAnsatz:
    """Class that creates and stores a symbolic ansatz circuit and can be used to
    produce instances of the circuit U(x)|0> for given parameters.

    Attributes:
        ansatz_circ: The symbolic circuit to be used as ansatz.
        feature_symbol_list: The list of symbols in the circuit, each corresponding to a feature.
    """
    def __init__(self, num_qubits: int, reps: int, gamma: float, entanglement_map: list[tuple[int, int]], hadamard_init: bool=True,
        onebpaulis: list[Pauli]=[Pauli.Z], twobpaulis: list[tuple[Pauli]]=[(Pauli.X,Pauli.X)]):
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
            onebpaulis: single-qubit Pauli operators for the circuit
            twobpaulis: two-qubit Pauli operators for the circuit
        """

        self.one_q_symbol_list = []
        self.two_q_symbol_list = []

        self.ansatz_circ = Circuit(num_qubits)
        self.feature_symbol_list = [Symbol('f_'+str(i)) for i in range(num_qubits)]

        if hadamard_init:
            for i in range(num_qubits):
                self.ansatz_circ.H(i)

        for _ in range(reps):
            for pauli in onebpaulis:
                for i in range(num_qubits):
                    exponent = (1/np.pi)*gamma*self.feature_symbol_list[i]
                    self.ansatz_circ.add_pauliexpbox(
                        PauliExpBox([pauli], exponent), qubits=[i]
                    )

            for (pauli0, pauli1) in twobpaulis:
                for (q0, q1) in entanglement_map:
                    symb0 = self.feature_symbol_list[q0]
                    symb1 = self.feature_symbol_list[q1]
                    exponent = gamma*gamma*(1 - symb0)*(1 - symb1)
                    self.ansatz_circ.add_pauliexpbox(
                        PauliExpBox([pauli0, pauli1], exponent), qubits=[q0, q1]
                    )

    def circuit_for_data(self, feature_values: list[float]) -> Circuit:
        """Produce the circuit with its symbols being replaced by the given values.
        """
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError("The number of values must match the number of symbols.")

        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)

        return the_circuit


def build_kernel_matrix(ansatz: KernelStateAnsatz, X, Y=None, info_file=None, mpi_comm=None) -> np.ndarray:
    """Use MPI to parallelise the calculation of entries of the kernel matrix.

    Notes:
        For matters of efficiency, it is assumed that `len(Y) <= len(X)`. If this is not
        the case, consider swapping them and applying a conjugate transposition to the output.

        By default, if `Y` is not provided, it is set to `X`. Using this default option when
        possible is preferable for matters of efficiency.

    Args:
        ansatz: a symbolic circuit describing the ansatz.
        X: A 2D array where `X[i, :]` corresponds to the i-th data point and
            each `X[:, j]` corresponds to the values of the j-th feature across
            all data points.
        Y: A 2D array where `Y[i, :]` corresponds to the i-th data point and
            each `Y[:, j]` corresponds to the values of the j-th feature across
            all data points. If not provided it is set to be equal to `X`.
        info_file: The name of the file where to save performance information of this call.
            If not provided, the performance information will only appear in stdout.
        mpi_comm: The MPI communicator created by the caller of this function. This
            function will attempt to parallelise across all processes within the
            communicator.

    Returns:
        A kernel matrix of dimensions `len(X)`x`len(Y)`.
    """

    # MPI information
    root = 0
    rank, n_procs = mpi_comm.Get_rank(), mpi_comm.Get_size()
    device_id = rank % getDeviceCount()

    # Dictionary to keep track of profiling information
    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (len(X), "entries")
        profiling_dict["lenY"] = (None if Y is None else len(Y), "entries")
        start_time = MPI.Wtime()

    # We use different parallelisation strategies depending on whether Y set to X or not

    ###########################################
    # Calculate the kernel matrix when X == Y #
    ###########################################
    if Y is None:
        time0 = MPI.Wtime()

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
            time1 = MPI.Wtime()
            print(f"[Rank 0] Circuit list generated. Time taken: {round(time1-time0,2)} seconds.")
            profiling_dict["r0_circ_gen"] = (time1-time0, "seconds")
            print("\nContracting the MPS of the circuits...")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # Contract the MPS of each of the circuits in this process
        this_proc_mps = []
        with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process
            progress_bar = 0
            progress_checkpoint = int(np.ceil(len(this_proc_circs) / 10))

            for k, circ in enumerate(this_proc_circs):
                # Simulate the circuit and obtain the output state as an MPS
                if circ is not None:
                    mps = simulate(libhandle, circ, ContractionAlg.MPSxGate)
                else:
                    mps = None
                this_proc_mps.append(mps)

                if rank == root and progress_bar * progress_checkpoint < k:
                    print(f"{progress_bar*10}%")
                    sys.stdout.flush()
                    progress_bar += 1

        if rank == root:
            time1 = MPI.Wtime()
            print("100%")
            duration = time1-time0
            print(f"[Rank 0] MPS contracted. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_circ_sim"] = (duration, "seconds")
            average = duration / circs_per_proc
            print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
            profiling_dict["avg_circ_sim"] = (average, "seconds")

            mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in this_proc_mps]
            total_bytes = sum(mps_byte_size) / (1024**2)
            print(f"[Rank 0] Total memory currently used: {round(total_bytes,2)} MiB")
            avg_bytes_per_mps = total_bytes / len(mps_byte_size)
            print(f"\tAverage MPS memory footprint: {round(avg_bytes_per_mps,2)} MiB")
            profiling_dict["avg_mps_mem"] = (avg_bytes_per_mps, "MiB")

            print("\nBroadcasting the MPS of the circuits.")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # Gather data from all processes into a list of lists `gathered_mps`.
        # For any i and r, `gathered_mps[i][r]` will contain the i-th entry
        # of `this_proc_mps` from the process with rank==r.
        gathered_mps = []
        for mps in this_proc_mps:  # Assumes all len(this_proc_mps) are the same
            gathered_mps.append(mpi_comm.allgather(mps))

        # Flatten the list of lists into `mps_list` so that we have all
        # entries of rank 0 first, then rank 1, etc. Explicitly:
        # `mps_list[i+r*circs_per_proc]` i-th entry of `this_proc_mps`
        # from rank==r.
        mps_list = [i_list[r] for r in range(n_procs) for i_list in gathered_mps]
        # Remove any `None` previously introduced for padding
        mps_list = [mps for mps in mps_list if mps is not None]

        if rank == root:
            time1 = MPI.Wtime()
            print(f"[Rank 0] MPS broadcasted in {round(time1-time0,2)} seconds")
            profiling_dict["r0_broadcast"] = (time1-time0, "seconds")

            mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in mps_list]
            total_bytes = sum(mps_byte_size) / (1024**2)
            print(f"Total MPS memory used per GPU: {round(total_bytes,2)} MiB")
            profiling_dict["gpu_mps_mem"] = (total_bytes, "MiB")

        print("Creating all random MPS on GPU device...")
        for mps in mps_list:
            mps.tensors = [cp.random.random(t.shape) for t in mps.tensors]

        # Enumerate all pairs of circuits to be overlapped
        pairs = [(i, j) for i in range(n_circs) for j in range(n_circs) if i < j]

        # Allocate space for kernel matrix
        kernel_mat = np.zeros(shape=(n_circs, n_circs))

        if rank == root:
            print("\nObtaining inner products...")
            time0 = MPI.Wtime()

        # Parallelise across all available processes
        with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process
            for mps in mps_list:
                mps.update_libhandle(libhandle)

            pairs_per_proc = int(np.ceil(len(pairs) / n_procs))
            progress_bar, progress_checkpoint = 0, int(np.ceil(pairs_per_proc / 10))
            for k in range(rank*pairs_per_proc, (rank+1)*pairs_per_proc):
                if k >= len(pairs): break

                # Run contraction
                (i, j) = pairs[k]
                mps0 = mps_list[i]
                mps1 = mps_list[j]

                timea = MPI.Wtime()
                overlap = mps0.vdot(mps1)
                kernel_mat[i, j] = kernel_mat[j, i] = (overlap*np.conj(overlap)).real
                timeb = MPI.Wtime()
                print(f"[{MPI.Wtime()}] Took {timeb - timea} seconds on process {rank}, device {device_id}.")


                # Report back to user
                if rank == root and progress_bar * progress_checkpoint < k:
                    print(f"{progress_bar*10}%")
                    sys.stdout.flush()
                    progress_bar += 1

        if rank == root:
            time1 = MPI.Wtime()
            print("100%")
            duration = time1 - time0
            print(f"[Rank 0] Inner products calculated. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_product"] = (duration, "seconds")
            average = duration / pairs_per_proc
            print(f"\tAverage time per inner product: {round(average,4)} seconds.\n")
            profiling_dict["avg_product"] = (average, "seconds")
            np.fill_diagonal(kernel_mat,1)

        # Collect all entries in the same process
        kernel_mat = mpi_comm.reduce(kernel_mat, op=MPI.SUM, root=root)

    ###########################################
    # Calculate the kernel matrix when X != Y #
    ###########################################
    else:
        time0 = MPI.Wtime()

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
            time1 = MPI.Wtime()
            print(f"[Rank 0] Circuit list generated. Time taken: {time1-time0} seconds.")
            profiling_dict["r0_circ_gen"] = (time1-time0, "seconds")
            print("\nContracting the MPS of the circuits...")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # Contract the MPS of each of the circuits in this process
        this_proc_x_mps = []
        this_proc_y_mps = []
        with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process
            progress_bar = 0
            this_proc_circs = this_proc_x_circs + this_proc_y_circs
            progress_checkpoint = int(np.ceil(len(this_proc_circs) / 10))

            for k, circ in enumerate(this_proc_circs):
                # Simulate the circuit and obtain the output state as an MPS
                if circ is not None:
                    mps = simulate(libhandle, circ, ContractionAlg.MPSxGate)
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
            time1 = MPI.Wtime()
            print("100%")
            duration = time1-time0
            print(f"[Rank 0]  MPS contracted. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_circ_sim"] = (duration, "seconds")
            average = duration / (x_circs_per_proc + y_circs_per_proc)
            print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
            profiling_dict["avg_circ_sim"] = (average, "seconds")

            mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in this_proc_x_mps + this_proc_y_mps]
            total_bytes = sum(mps_byte_size) / (1024**2)
            print(f"[Rank 0] Total memory currently used: {round(total_bytes,2)} MiB")
            avg_bytes_per_mps = total_bytes / len(mps_byte_size)
            print(f"\tAverage MPS memory footprint: {round(avg_bytes_per_mps,2)} MiB")
            profiling_dict["avg_mps_mem"] = (avg_bytes_per_mps, "MiB")

            print("\nBroadcasting the MPS of the circuits.")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # The MPS from the X dataset need not be broadcasted, those from Y do.
        # Remove any `None` previously introduced for padding
        this_proc_x_mps = [mps for mps in this_proc_x_mps if mps is not None]

        # Gather data from all processes into a list of lists `gathered_mps`.
        # For any i and r, `gathered_mps[i][r]` will contain the i-th entry
        # of `this_proc_y_mps` from the process with rank==r.
        gathered_mps = []
        for mps in this_proc_y_mps:  # Assumes all len(this_proc_y_mps) are the same
            gathered_mps.append(mpi_comm.allgather(mps))

        # Flatten the list of lists into `y_mps_list` so that we have all
        # entries of rank 0 first, then rank 1, etc. Explicitly:
        # `y_mps_list[i+r*circs_per_proc]` i-th entry of `this_proc_y_mps`
        # from rank==r.
        y_mps_list = [i_list[r] for r in range(n_procs) for i_list in gathered_mps]
        # Remove any `None` previously introduced for padding
        y_mps_list = [mps for mps in y_mps_list if mps is not None]

        if rank == root:
            time1 = MPI.Wtime()
            print(f"[Rank 0] MPS broadcasted in {round(time1-time0,2)} seconds")
            profiling_dict["r0_broadcast"] = (time1-time0, "seconds")

            mps_byte_size = [sum(t.nbytes for t in mps.tensors) for mps in this_proc_x_mps + y_mps_list]
            total_bytes = sum(mps_byte_size) / (1024**2)
            print(f"Total MPS memory used per GPU: {round(total_bytes,2)} MiB")
            profiling_dict["gpu_mps_mem"] = (total_bytes, "MiB")

        print("Creating all random MPS on GPU device...")
        for mps in y_mps_list:
            mps.tensors = [cp.random.random(t.shape) for t in mps.tensors]
        for mps in this_proc_x_mps:
            mps.tensors = [cp.random.random(t.shape) for t in mps.tensors]

        # Allocate space for kernel matrix
        kernel_mat = np.zeros(shape=(x_circs, y_circs))

        if rank == root:
            print("\nObtaining inner products...")
            time0 = MPI.Wtime()

        # Each process will calculate the inner products of its MPS from X with
        # all of the MPS from Y.
        progress_bar = 0
        progress_checkpoint = int(np.ceil(len(this_proc_x_mps) / 10))

        with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process

            for y_mps in y_mps_list:
                y_mps.update_libhandle(libhandle)

            for i, x_mps in enumerate(this_proc_x_mps):
                x_mps.update_libhandle(libhandle)

                for j, y_mps in enumerate(y_mps_list):

                    timea = MPI.Wtime()
                    overlap = x_mps.vdot(y_mps)
                    kernel_mat[i + rank*x_circs_per_proc, j] = (overlap*np.conj(overlap)).real
                    timeb = MPI.Wtime()
                    print(f"[{MPI.Wtime()}] Took {timeb - timea} seconds on process {rank}, device {device_id}.")

                    if rank == root and progress_bar * progress_checkpoint < i:
                        print(f"{progress_bar*10}%")
                        sys.stdout.flush()
                        progress_bar += 1

        if rank == root:
            time1 = MPI.Wtime()
            print("100%")
            duration = time1 - time0
            print(f"[Rank 0] Inner products calculated. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_product"] = (duration, "seconds")
            average = duration / (len(this_proc_x_mps) * len(y_mps_list))
            print(f"\tAverage time per inner product: {round(average,4)} seconds.\n")
            profiling_dict["avg_product"] = (average, "seconds")

        # Collect all entries in the same process
        kernel_mat = mpi_comm.reduce(kernel_mat, op=MPI.SUM, root=root)

    if rank == root:
        end_time = MPI.Wtime()
        profiling_dict["total_time"] = (end_time-start_time, "seconds")

    # If requested by user, dump `profiling_dict` to file
    if info_file is not None and rank == root:
        with open(info_file, 'w') as fp:
            json.dump(profiling_dict, fp, indent=4)

    return kernel_mat
