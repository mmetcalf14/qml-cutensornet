# Scripts to reproduce figure

The reproduction of Figure 5 and Table 1 is achieved in three steps:

- T1. Execute `./run_all.sh`. This will generate a list of files `train_*.json` in a `raw/` folder in this directory. A copy of these is provided in `raw.zip`.
- T2. Execute `python to_csv.py`. This will generate two files `gpu_results.csv` and `cpu_results.csv` from the contents of `raw/`.
- T3. Execute `python plot.py`. This will use these `*_results.csv` files to generate the two subfigures of Figure 5, which will pop-up in a new window. Table I will be printed in the command line.
