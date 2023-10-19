"""
This example showcases the use of MPI to run an embarrasingly parallel task on multiple
GPUs. The task is to find the inner products of all pairs of ``n_circ`` circuits; each
inner product can be computed separately from the rest, hence the parallelism.
All of the circuits in this example are defined in terms of the same symbolic circuit
``sym_circ`` on ``n_qubits``, with the symbols taking random values for each circuit.

This is one script in a set of three scripts (all named ``mpi_overlap_bcast_*``)
where the difference is which object is broadcasted between the processes. These
are ordered from most efficient to least efficient:
- ``mpi_overlap_bcast_mps.py`` broadcasts ``MPS``.
- ``mpi_overlap_bcast_net.py`` broadcasts ``TensorNetwork``.
- ``mpi_overlap_bcast_circ.py`` broadcasts ``pytket.Circuit``.

In the present script, we proceed as follows:
- Create the same symbolic circuit on every process
- Each process creates a fraction of the ``n_circs`` instances of the symbolic circuit.
    - Then, converts each of the circuits it generated to a ``TensorNetwork`` object.
- Broadcast these ``TensorNetwork`` objects to all other processes.
- Find an efficient contraction path for the inner product TN.
    - Since all circuits have the same structure, the same path can be used for all.
- Distribute calculation of inner products uniformly accross processes. Each process:
    - Creates a TN representing the inner product ``<0|C_i^dagger C_j|0>``
    - Contracts the TN using the contraction path previously found.

The script is able to run on any number of processes; each process must have access to
a GPU of its own.

Notes:
    - We used a very shallow circuit with low entanglement so that contraction time is
      short. Other circuits may be used with varying cost in runtime and memory.
    - Here we are using ``cq.contract`` directly (i.e. cuTensorNet API), but other
      functionalities from our extension (and the backend itself) could be used
      in a similar script.
"""

import sys
from random import random

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
import cuquantum as cq

from pytket.circuit import Circuit, fresh_symbol

from pytket.extensions.cutensornet import TensorNetwork

# Parameters
n_qubits = 100
n_circs = 32

root = 0
comm = MPI.COMM_WORLD

rank, n_procs = comm.Get_rank(), comm.Get_size()
# Assign GPUs uniformly to processes
device_id = rank % getDeviceCount()

time_start = MPI.Wtime()
net_list = []

if n_circs % n_procs != 0:
    raise RuntimeError(
        "Current version requires that n_circs is a multiple of n_procs."
    )

#if rank == root:
#    print("\nGenerating the circuits.")
#    time0 = MPI.Wtime()

# Generate the list of circuits in parallel
circs_per_proc = n_circs // n_procs
this_proc_circs = []

# Generate the symbolic circuit
sym_circ = Circuit(n_qubits)
even_qs = sym_circ.qubits[0::2]
odd_qs = sym_circ.qubits[1::2]

for q0, q1 in zip(even_qs, odd_qs):
    sym_circ.TK2(fresh_symbol(), fresh_symbol(), fresh_symbol(), q0, q1)
for q in sym_circ.qubits:
    sym_circ.H(q)
for q0, q1 in zip(even_qs[1:], odd_qs):
    sym_circ.TK2(fresh_symbol(), fresh_symbol(), fresh_symbol(), q0, q1)
free_symbols = sym_circ.free_symbols()

# Create each of the circuits
for _ in range(circs_per_proc):
    symbol_map = {symbol: random() for symbol in free_symbols}
    my_circ = sym_circ.copy()
    my_circ.symbol_substitution(symbol_map)
    this_proc_circs.append(my_circ)

#if rank == root:
#    time1 = MPI.Wtime()
#    print(f"Circuit list generated. Time taken: {time1-time0} seconds.\n")
#    print("Converting circuits to nets.")
#    sys.stdout.flush()
#    time0 = MPI.Wtime()

# Convert the circuit to a TensorNetwork for of each of the circuits in this process
this_proc_nets = [TensorNetwork(circ) for circ in this_proc_circs]

#if rank == root:
#    time1 = MPI.Wtime()
#    print(f"Net list generated. Time taken: {time1-time0} seconds.\n")
#    print("Broadcasting the net of the circuits.")
#    sys.stdout.flush()

# Broadcast the list of nets
time0 = MPI.Wtime()
for proc_i in range(n_procs):
    net_list += comm.bcast(this_proc_nets, proc_i)

time1 = MPI.Wtime()
#print(f"Nets broadcasted to {rank} in {time1-time0} seconds")

# Enumerate all pairs of circuits to be calculated
pairs = [(i, j) for i in range(n_circs) for j in range(n_circs) if i < j]

# Find an efficient contraction path to be used by all contractions
time0 = MPI.Wtime()
# Prepare the Network object
net0 = net_list[0]  # Since all circuits have the same structure
net1 = net_list[1]  # we use these two as a template
overlap_network = cq.Network(*net0.vdot(net1), options={"device_id": device_id})
# Compute the path on each process with 8 samples for hyperoptimization
path, info = overlap_network.contract_path(optimize={"samples": 8})
# Select the best path from all ranks.
opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
if rank == root:
    time0 = MPI.Wtime()
#    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")
# Broadcast path from the sender to all other processes
path = comm.bcast(path, sender)
# Report back to user
duration_bcast = MPI.Wtime() - time0
#if rank == root:
#    print(f"Contraction path found in {time1-time0} seconds.\n")
#    sys.stdout.flush()

# Parallelise across all available processes

time0 = MPI.Wtime()

iterations, remainder = len(pairs) // n_procs, len(pairs) % n_procs
progress_bar, progress_checkpoint = 0, iterations // 10
for k in range(iterations):
    # Run contraction
    (i, j) = pairs[k * n_procs + rank]
    net0 = net_list[i]
    net1 = net_list[j]
    timea = MPI.Wtime()
    overlap = cq.contract(
        *net0.vdot(net1), options={"device_id": device_id}, optimize={"path": path}
    )
    timeb=MPI.Wtime()
    print('This iteration took {}s for rank {}'.format(timeb-timea,rank))
    # Report back to user
    # print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")
    if rank == root and progress_bar * progress_checkpoint < k:
#        print(f"{progress_bar*10}%")
#        sys.stdout.flush()
        progress_bar += 1

if rank < remainder:
    # Run contraction
    (i, j) = pairs[iterations * n_procs + rank]
    net0 = net_list[i]
    net1 = net_list[j]
    overlap = cq.contract(
        *net0.vdot(net1), options={"device_id": device_id}, optimize={"path": path}
    )
    # Report back to user
    # print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")

time1 = MPI.Wtime()
time_end = MPI.Wtime()

# Report back to user
duration = time1 - time0
print(f"Runtime at {rank} is {duration}")
print(f"Process {rank} has computed inner products. \n\tAt GPU {device_id}. \n\tCalculation took {duration} seconds. \n\tStarted at {time0}. \n\tBroadcast time {duration_bcast} seconds.")

totaltime = comm.reduce(duration, op=MPI.MAX, root=root)
if rank == root:
    print(f"\n**Walltime duration** {totaltime} seconds\n")
