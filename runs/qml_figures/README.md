# Scripts to reproduce figure

The reproduction of Figures 9 and 10 is achieved in two steps:

- T1. Execute `./run_all.sh`. This will run multiple slurm jobs, each taking the appropriate number of GPUs. The script is designed to run on Perlmutter using the setup described in the README at the root of this repository. Other computers or setups may require changes to the scripts in the directory `slurm_scripts`. When successful, this will generate a list of files `*.npy` in a `raw/` folder in this directory. A copy of these is provided in `raw.zip`.
- T2. Execute `python plot.py`. This will use contents of the `raw/` folder to generate both figures, which will pop-up in a new window (one after the other).
