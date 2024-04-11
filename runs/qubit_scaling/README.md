# Scripts to reproduce figure

The reproduction of Figure 7 is achieved in three steps:

- T1. Execute `./run_all.sh`. This will generate a list of files `train_*.json` on the root directory of this repository. A copy of these is provided in `data.zip`.
- T2. Move all `train_*.json` files to a folder `data/` within this directory and execute `python to_csv.py`. This will generate a `results.csv` file.
- T3. Execute `python plot.py`. This will use the `results.csv` file to generate the figure, which will pop-up in a new window.
