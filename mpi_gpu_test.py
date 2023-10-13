from mpi4py import MPI
from cupy.cuda.runtime import getDeviceCount

mpi_comm = MPI.COMM_WORLD
rank, n_procs = mpi_comm.Get_rank(), mpi_comm.Get_size()
root = 0

device_id = rank % getDeviceCount()

print(f"Hi, I am process {rank} and I see {getDeviceCount()} GPUs. I'm using GPU {device_id}.")


