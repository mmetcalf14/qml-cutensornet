# Scripts to reproduce table

The reproduction of the values of Table II.

- T1. Execute `./run_all.sh`. This will generate a list of files `*.npy` in a `raw/` folder in this directory, both for the Gaussian kernel (in `raw/gaussian/`) and the quantum kernel (`raw/quantum/`). For the Gaussian kernel it makes use of `classical_main.py` which runs SVC with a default classical RBF (Gaussian) kernel using `scikit-learn`. See [scikit-learn docs](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#rbf-kernel).
- T2. Execute `to_csv.py`. This will read the files in the `raw/` folder, process their results and output a `results.csv` file with the values reported in Table II.
