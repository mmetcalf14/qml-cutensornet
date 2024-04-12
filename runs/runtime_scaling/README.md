# Scripts to reproduce figure

The reproduction of Figure 8 is achieved in three steps:

- T1. Execute `./run_all.sh`. This will run multiple slurm jobs, each taking the appropriate number of GPUs. The script is designed to run on Perlmutter using the setup described in the README at the root of this repository. Other computers or setups may require changes to the scripts in the directory `slurm_scripts`. When successful, this will generate a list of files `train_*.json` in a `raw/` folder in this directory. A copy of these is provided in `raw.zip`.
- T2. Execute `python to_csv.py`. This will generate a `results.csv` file from the contents of `raw/`.
- T3. Execute `python plot.py`. This will use the `results.csv` file to generate the figure, which will pop-up in a new window.
