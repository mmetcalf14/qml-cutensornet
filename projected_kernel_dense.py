import sys
import json
import pathlib

from mpi4py import MPI

import numpy as np
from sympy import Symbol
from statistics import mean, median

from pytket import Circuit
from pytket.transform import Transform
from pytket.architecture import Architecture
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit
from pytket.circuit import PauliExpBox, Pauli

from quimb_mps_1rdm import Config, simulate, compute_1_rdm, build_ham_mpo_operator
#import quimb as qt



class ProjectedKernelStateAnsatz:
    """Class that creates and stores a symbolic ansatz circuit and can be used to
    produce instances of the circuit U(x)|0> for given parameters.

    Attributes:
        ansatz_circ: The symbolic circuit to be used as ansatz.
        feature_symbol_list: The list of symbols in the circuit, each corresponding to a feature.
    """
    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        reps: int,
        gamma: float,
        entanglement_map: list[tuple[int, int]],
        hadamard_init: bool=True
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
        self.feature_symbol_list = [Symbol('f_'+str(i)) for i in range(num_features)]

        if hadamard_init:
            for i in range(num_qubits):
                self.ansatz_circ.H(i)

#        for _ in range(reps):
#            counterf = 0
#            counterq = 0
#            it = 0
#            for i in range(2*num_features):
#                exponent = (1/np.pi)*gamma*self.feature_symbol_list[counterf]
#                if it % 2 == 0:
#                    self.ansatz_circ.Rz(exponent, counterq)
#                else: self.ansatz_circ.Ry(exponent, counterq)
#                
#                if counterf < num_features:
#                    counterf += 1
#                else: counterf = 0
#                
#                if counterq < num_qubits:
#                    counterq += 1
#                else: counterq = 0
#                
#                if it%num_qubits == 0:
#                    it+=1
#                
#
#            for (q0, q1) in entanglement_map:
#                symb0 = self.feature_symbol_list[q0]
#                symb1 = self.feature_symbol_list[q1]
#                exponent = gamma*gamma*(1 - symb0)*(1 - symb1)
#                exponent = 1
#                self.ansatz_circ.YYPhase(exponent, q0, q1)

        # Apply TKET routing to compile circuit to line architecture
        for _ in range(reps):
            for i in range(num_qubits):
                exponent = (1/np.pi)*gamma*self.feature_symbol_list[i]
                self.ansatz_circ.Rz(exponent, i)

            for (q0, q1) in entanglement_map:
                symb0 = self.feature_symbol_list[q0]
                symb1 = self.feature_symbol_list[q1]
                exponent = gamma*gamma*(1 - symb0)*(1 - symb1)
                self.ansatz_circ.XXPhase(exponent, q0, q1)
        cu = CompilationUnit(self.ansatz_circ)
        architecture = Architecture(
            [(i, i + 1) for i in range(self.ansatz_circ.n_qubits - 1)]
        )
        DefaultMappingPass(architecture).apply(cu)
        self.ansatz_circ = cu.circuit
        Transform.DecomposeBRIDGE().apply(self.ansatz_circ)


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
        mpi_comm,
        config: Config,
        ansatz: ProjectedKernelStateAnsatz,
        X,
        Y=None,
        alpha=1,
        info_file=None,
        cpu_max_mem=6,
        minutes_per_checkpoint=None,
    ) -> np.ndarray:
    """Calculation of entries of the kernel matrix.

    Notes:
        For matters of efficiency, it is assumed that `len(Y) <= len(X)`. If this is not
        the case, consider swapping them and applying a conjugate transposition to the output.

        By default, if `Y` is not provided, it is set to `X`. Using this default option when
        possible is preferable for matters of efficiency.

    Args:
        mpi_comm: The MPI communicator created by the caller of this function. This
            function will attempt to parallelise across all processes within the
            communicator.
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
        cpu_max_mem: The number of GB available in each CPU. You should leave some
            margin for the background tasks running on the computer. Defaults to 6 GB.
        minutes_per_checkpoint: The amount of time (in minutes) elapsed between different
            checkpoint saves of the kernel matrix. If None, no checkpoints are saved.

    Returns:
        A kernel matrix of dimensions `len(Y)`x`len(X)`.

    Raises:
        ValueError: It is assumed that `len(Y) <= len(X)`. If this is not the case,
        an error is raised. You will need to swap the inputs and transpose the output.
    """
    if Y is not None and len(X) < len(Y):
        raise ValueError("X must not be smaller than Y. Swap input order and transpose output.")
    n_qubits = ansatz.ansatz_circ.n_qubits

    # MPI information
    root = 0
    rank = mpi_comm.Get_rank()
    n_procs = mpi_comm.Get_size()

    entries_per_chunk = int(np.ceil(len(X) / n_procs))
    max_mps_per_cpu = 2*entries_per_chunk  # X + Y chunks
    
    mpo_matrix = build_ham_mpo_operator(n_qubits)

    # Checkpointing file
    pathlib.Path("tmp").mkdir(exist_ok=True)
    checkpoint_file = pathlib.Path(f"tmp/checkpoint_rank_{rank}_" + info_file + ".npy")

    # Dictionary to keep track of profiling information
    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (len(X), "entries")
        profiling_dict["lenY"] = (None if Y is None else len(Y), "entries")
        start_time = MPI.Wtime()

    # Divide X into chunks and generate circuits
    x_chunks = n_procs  # As many chunks as CPUs
    # Tiles will always be squares of width at most `entries_per_chunk`.
    # If it is not completely filled there will be some `None` entries at the end.
    circs_x_chunk = [None]*entries_per_chunk
    
    for i in range(entries_per_chunk):
        offset = rank * entries_per_chunk
        if i + offset < len(X):
            circs_x_chunk[i] = ansatz.circuit_for_data(X[i+offset, :])

    # Divide Y into chunks
    if Y is not None:  # If Y != X
        y_chunks = int(np.ceil(len(Y) / entries_per_chunk))
    else:  # If Y == X
        y_chunks = x_chunks

    # The largest multiple of `y_chunks` that is smaller than `n_procs` will be the
    # number of CPUs communicating in round robin.
    n_procs_in_RR = n_procs - (x_chunks % y_chunks)

    # Generate the circuis of the Y chunk (if Y != X)
    if Y is not None:
        # Tiles will always be squares of width at most `entries_per_chunk`.
        # If it is not completely filled there will be some `None` entries at the end.
        circs_y_chunk = [None]*entries_per_chunk

        # Only generate the circuits for the CPUs that participate in round robin
        if rank < n_procs_in_RR:
            offset = (rank % y_chunks) * entries_per_chunk
            for i in range(entries_per_chunk):
                if i + offset < len(Y):
                    circs_y_chunk[i] = ansatz.circuit_for_data(Y[i+offset, :])
        # The CPUs with `rank >= n_procs_in_RR` do not fit in the round robin, so
        # they will be receiving their Y MPS chunk from the first few CPUs.

    # Report back to user
    if rank == root:
        duration = MPI.Wtime() - start_time
        print(f"[Rank 0] Circuit list generated. Time taken: {round(duration,2)} seconds.")
        profiling_dict["r0_circ_gen"] = [duration, "seconds"]
        print("\nContracting the MPS of the circuits from the X dataset...")
        sys.stdout.flush()

    # Each CPU contracts the MPS from its X chunk
    rdm_x_chunk = []
    rdm_x_time = []
    progress_bar = 0
    progress_tick = int(np.ceil(entries_per_chunk / 10))

    for k, circ in enumerate(circs_x_chunk):
        # Simulate the circuit and obtain the output state as an MPS
        if circ is not None:
            time0 = MPI.Wtime()
            #mps = simulate(circ, config)
            rdm = compute_1_rdm(circ, config, n_qubits, mpo_matrix)
            rdm_x_time.append(MPI.Wtime() - time0)
        else:
            rdm = None
        rdm_x_chunk.append(rdm)

        if rank == root and progress_bar * progress_tick < k:
            print(f"{progress_bar*10}%")
            sys.stdout.flush()
            progress_bar += 1

    # Report back to user
    if rank == root:
        print("100%")
        duration = sum(rdm_x_time)
        print(f"[Rank 0] MPS of chunk X contracted. Time taken: {round(duration,2)} seconds.")
        profiling_dict["r0_circ_sim"] = [duration, "seconds"]
        average = mean(rdm_x_time)
        print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
        profiling_dict["avg_circ_sim"] = [average, "seconds"]
        profiling_dict["median_circ_sim"] = [median(rdm_x_time), "seconds"]

        if Y is not None:
            print("\nContracting the MPS of the circuits from the Y dataset...")
        sys.stdout.flush()

    # Each CPU contracts the MPS from its Y chunk (only if Y != X)
    if Y is not None:
        rdm_y_chunk = []
        rdm_y_time = []
        progress_bar = 0
        progress_tick = int(np.ceil(entries_per_chunk / 10))

        for k, circ in enumerate(circs_y_chunk):
            # Simulate the circuit and obtain the output state as an MPS
            if circ is not None:
                time0 = MPI.Wtime()
                #mps = simulate(circ, config)
                rdm = compute_1_rdm(circ, config, n_qubits, mpo_matrix)
                rdm_y_time.append(MPI.Wtime() - time0)
            else:
                rdm = None
            rdm_y_chunk.append(rdm)

            if rank == root and progress_bar * progress_tick < k:
                print(f"{progress_bar*10}%")
                sys.stdout.flush()
                progress_bar += 1

        # Report back to user
        if rank == root:
            print("100%")
            duration = sum(rdm_y_time)
            print(f"[Rank 0] MPS of chunk Y contracted. Time taken: {round(duration,2)} seconds.")
            profiling_dict["r0_circ_sim"][0] += duration
            average = mean(rdm_x_time + rdm_y_time)
            print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
            profiling_dict["avg_circ_sim"] = [average, "seconds"]
            profiling_dict["median_circ_sim"] = [median(rdm_x_time + rdm_y_time), "seconds"]

    # If Y == X then Y chunk for the first iteration will be a copy of the X chunk
    else:
        rdm_y_chunk = [rdm.copy() if rdm is not None else None for rdm in rdm_x_chunk]

    # Report back to user
    if rank == root:
        print("\nFinished contracting all MPS.")

        print("\nCalculating kernel matrix...")
        profiling_dict["r_nonRR_recv"] = [0, "seconds"]
        profiling_dict["r0_RR_recv"] = [0, "seconds"]
        sys.stdout.flush()
        tiling_start_time = MPI.Wtime()

    # Try to recover from the last checkpoint (if any)
    if checkpoint_file.is_file():
        # Load the kernel matrix from the checkpoint file (one per process)
        kernel_mat = np.load(checkpoint_file)
        print(f"[Rank {rank}] Recovered from checkpoint!")
        sys.stdout.flush()
    else:
        # Allocate space for kernel matrix
        len_Y = len(Y) if Y is not None else len(X)
        kernel_mat = np.zeros(shape=(len_Y, len(X)))
    last_checkpoint_time = MPI.Wtime()

    vdot_time = []
    # Compute tiles of the kernel-matrix and pass the Y chunks around in round robin
    if Y is not None:
        iterations = y_chunks
    else:
        iterations = (x_chunks // 2) + 1  # Some iterations are skipped thanks to symmetry
    for this_iteration in range(iterations):

        if rank == root:
            print(f"\n\tBegin next collection of tiles... ({this_iteration+1}/{iterations})")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # The last few CPUs don't fit in the round robin, so they fetch their MPS now
        for proc_send in range(x_chunks % y_chunks):
            proc_recv = n_procs_in_RR + proc_send  # `proc_recv` doesn't fit in round robin

            # MPI specs guarantess msg order is preserved.
            if rank == proc_send:
                for rdm in rdm_y_chunk:
                    mpi_comm.send(rdm, dest=proc_recv)
            # Blocking receive, since `proc_recv` needs the MPS before continuing
            if rank == proc_recv:
                for i in range(entries_per_chunk):
                    rdm_y_chunk[i] = mpi_comm.recv(source=proc_send)

        if rank == root:
            duration = MPI.Wtime() - time0
            profiling_dict["r_nonRR_recv"][0] += duration
            print("\tCalculating inner products...")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # Each CPU calculates the inner products in its assigned tile of the kernel matrix
        progress_bar = 0
        progress_tick = int(np.ceil(entries_per_chunk / 10))

        for i, x_rdm in enumerate(rdm_x_chunk):
            if x_rdm is None: break  # Reached the padded entries; stop
            for j, y_rdm in enumerate(rdm_y_chunk):
                if y_rdm is None: break  # Reached the padded entries; stop
                
                x_index = i + entries_per_chunk*rank
                y_index = j + entries_per_chunk*((rank+this_iteration) % y_chunks)
#                if Y is None:
#                    if x_index == 5 and y_index == 0:
#                        print('Success on proc {}'.format(rank))
#                    else: print('not obtained')
                # Skip if this value was saved in the checkpoint
                if kernel_mat[y_index, x_index] != 0: break
                
                time_a = MPI.Wtime()
                proj_kern = sum(np.linalg.norm(x_rdm[i,:,:]-y_rdm[i,:,:])**2 for i in range(n_qubits))
                vdot_time.append(MPI.Wtime() - time_a)

                kernel_entry = np.exp(-1*alpha*proj_kern) #abs(overlap)**2
                kernel_mat[y_index, x_index] = kernel_entry
                # If X == Y, some entries can be filled thanks to symmetry
                if Y is None:
                    if not (
                        this_iteration == 0 or  # Don't do it for the block diagonal
                        x_chunks % 2 == 0 and this_iteration == iterations - 1  # Don't do for last iteration
                    ):
                        kernel_mat[x_index, y_index] = kernel_entry
#                     NOTE: We skip for the block diagonal since we calculate each entry of it as if it were
#                     a standard tile, so the symmetry would just rewrite values that we have calculated.
#                     NOTE: When `x_chunks` is even we must skip the last iteration. Otherwise two different CPUs
#                     would solve the same tile, causing these to have double their value after applying
#                     `mpi_comm.reduce` with SUM operator. When `x_chunks` is odd this is not an issue.

            if rank == root and progress_bar * progress_tick < i:
                print(f"\t{progress_bar*10}%")
                sys.stdout.flush()
                progress_bar += 1

        # Report back to user
        if rank == root:
            print("\t100%")
            duration = MPI.Wtime() - time0
            print(f"\t[Rank 0] Inner products calculated. Time taken: {round(duration,2)} seconds.")
            sys.stdout.flush()
            time0 = MPI.Wtime()

        # Save a checkpoint if it's time
        if minutes_per_checkpoint is not None and last_checkpoint_time + 60*minutes_per_checkpoint < MPI.Wtime():
            last_checkpoint_time = MPI.Wtime()

            # Remove the previous checkpoint file
            checkpoint_file.unlink(missing_ok=True)
            # Create a new checkpoint
            np.save(checkpoint_file, kernel_mat)
            # Inform user
            print(f"[Rank {rank}] Checkpoint saved at {checkpoint_file}!")
            sys.stdout.flush()

        # Perform message passing in round robin
        if rank < n_procs_in_RR:  # The last few CPUs don't participate
            # Each process sends its Y chunk to the next process in a cycle
            for i, rdm in enumerate(rdm_y_chunk):
                rdm_y_chunk[i] = mpi_comm.sendrecv(rdm, dest=(rank-1)%n_procs_in_RR)

        # Report back to user
        if rank == root:
            duration = MPI.Wtime() - time0
            print(f"\t[Rank 0] Round robin message passing completed in {round(duration,2)} seconds")
            sys.stdout.flush()
            profiling_dict["r0_RR_recv"][0] += duration
    # Collect the tiles of all CPUs into a the final kernel matrix
    kernel_mat = mpi_comm.reduce(kernel_mat, op=MPI.SUM, root=root)

    # Report back to user
    if rank == root:
        tiling_duration = MPI.Wtime() - tiling_start_time
        total_duration = MPI.Wtime() - start_time
        profiling_dict["kernel_mat_time"] = [tiling_duration, "seconds"]
        profiling_dict["total_time"] = [total_duration, "seconds"]
        profiling_dict["r0_product"] = [sum(vdot_time), "seconds"]

        print("\nFinished calculating all inner products.")
        average = mean(vdot_time)
        print(f"\tAverage time per inner product: {round(average,4)} seconds.")
        profiling_dict["avg_product"] = [average, "seconds"]
        profiling_dict["median_product"] = [median(vdot_time), "seconds"]
        print("")

        # If requested by user, dump `profiling_dict` to file
        if info_file is not None:
            with open(info_file + ".json", 'w') as fp:
                json.dump(profiling_dict, fp, indent=4)
            print(f"Profiling information saved at {info_file}.json.\n")
        sys.stdout.flush()

    # We can delete the checkpoint file (useful, so that we avoid risk of collisions)
    checkpoint_file.unlink(missing_ok=True)

    return kernel_mat
