# Scripts to reproduce figure

The reproduction of Figure 6 is achieved in two steps:

- T1. Execute `./run_all.sh`. This will generate a list of files `*.out` in folders `raw/d6` and `raw/d12` of this directory. A copy of these is provided in `data.zip`.
- T2. Execute `python plot.py`. This will use the data in the `raw/` folder to generate the figure, which will pop-up in a new window.
