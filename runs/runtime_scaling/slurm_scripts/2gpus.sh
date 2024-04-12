#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH --ntasks-per-node 2
#SBATCH -q regular
#SBATCH -t 00:30:00

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export MPI4PY_RC_RECV_MPROBE=0

cd ../..
srun python main_no_test.py $@
mv train_Nf* runs/runtime_scaling/raw/
