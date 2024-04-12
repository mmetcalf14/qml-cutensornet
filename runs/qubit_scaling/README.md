# Scripts to reproduce figure

The reproduction of Figure 7 is achieved in three steps:

- T1. Execute `./run_all.sh`. This will generate a list of files `train_*.json` in a `raw/` folder in this directory. A copy of these is provided in `raw.zip`.
- T2. Execute `python to_csv.py`. This will generate a `results.csv` file from the contents of `raw/`.
- T3. Execute `python plot.py`. This will use the `results.csv` file to generate the figure, which will pop-up in a new window.
