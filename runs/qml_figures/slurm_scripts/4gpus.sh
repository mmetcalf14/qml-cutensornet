#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH --ntasks-per-node 4
#SBATCH -q regular
#SBATCH -t 05:00:00

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export MPI4PY_RC_RECV_MPROBE=0

cd ../..
srun python main.py $@
mv data/* runs/qml_figures/raw/
