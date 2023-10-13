from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
rank, n_procs = mpi_comm.Get_rank(), mpi_comm.Get_size()
root = 0

print(f"Hi, I am process {rank}")


