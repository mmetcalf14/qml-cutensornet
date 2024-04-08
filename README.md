Code for our experiments on Quantum Kernel methods. Two backends are provided:
- *GPU*: simulations run on NVIDIA GPUs via [pytket-cutensornet](https://github.com/CQCL/pytket-cutensornet).
- *CPU*: simulations run on CPUs via [ITensors.jl](https://github.com/ITensor/ITensors.jl).

## Installation

This project uses `python>=3.10`. If you wish to use the GPU backend, we recommend you follow the `conda` installation described in the following section; then run the command below. Both backends can be set up in the same environment.

The common packages required by both backends can be installed via the following command:
```
pip install -r requirements.txt
```

**Note**: These installation instructions have been tested and can be reproduced in NERSC's Perlmutter and Google Cloud Platform. Set up on other platforms may vary.

### GPU backend

To use this backend you require a device with an NVIDIA GPU with Compute Capability +7.0 (check it [here](https://developer.nvidia.com/cuda-gpus)). Our package makes use of NVIDIA's `cuquantum-python`, and we recommend the `conda` installation detailed below (see [here](https://docs.nvidia.com/cuda/cuquantum/latest/python/README.html#installation) for more instructions).

Create the conda environment. In order to make use of CUDA-aware MPI (and assumming your system already contains a valid CUDA-aware `mpich`), we add the extra `"mpich=*=external_*"`.
```
conda create -n qml_env -c conda-forge python=3.10 cuquantum-python=23.10.0 "mpich=*=external_*"
conda activate qml_env
```
Install `mpi4py` to make use of the CUDA-aware MPI library.
```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

Install `pytket-cutensornet` version 0.6.0
```
pip install pytket-cutensornet==0.6.0
```
**Note**: Remember to run `pip install -r requirements.txt` after setting up the conda environment.

### CPU backend

This backend makes use of Julia, ITensors.jl, pyCall and a local package KernelPkg.jl.

To install *Julia*, follow: https://docs.julialang.org/en/v1/manual/getting-started/

To install *ITensors.jl*, enter the Julia REPL and press `]` to enter the package manager. Type `add ITensors`.

To install *pyCall*:
- `pip install julia`
- In Python REPL, `import julia` followed by `julia.install()`

Test pyCall installation was successful by entering the Python REPL and typing:
 ```
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Base
Base.sind(60)
 ```

Finally, to install *KernelPkg.jl*, move the contents of the folder `KernelPkg` in this repository to `~/.julia/dev/KernelPkg`. Then, open the Julia REPL and press `]` to enter the package manager. Locally install the package by typing:
`dev ~/.julia/dev/KernelPkg`.


## Running experiments

Experiments use the Elliptic Bitcoin Dataset, which needs to be [downloaded from Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and stored in the directory ``datasets/elliptic_bitcoin_dataset``. The two files in the directory should be ``elliptic_txs_classes.csv`` and ``elliptic_txs_features.csv``.
Then, run the following command to preprocess the dataset.
```
python elliptic_preproc.py
```

If installation was successful, you should be able to run experiments via the command:
```
python main.py <backend> <num_features> <layers> <gamma> <distance> <n_illicit> <n_licit> <data_seed> <data_file>
```
where each of the arguments have the following meaning:
- `<backend>` is either `"GPU"` or `"CPU"`.
- `<num_features>` is the number of features from the dataset to be used (which corresponds to the number of qubits used). This should be an integer.
- `<layers>` is the number of ansatz layers in the circuit. This should be an integer.
- `<gamma>` is the kernel bandwidth hyperparamter. This should be a real number between `0.0` and `1.0`.
- `<distance>` is the qubit interaction distance in the circuit ansatz. This should be an integer.
- `<n_illicit>` is the number of data points marked as "illicit" in the Elliptic Bitcoin Dataset to be used. The sum of `<n_illicit> + <n_licit>` determines the dataset size. This should be an integer.
- `<n_licit>` is the number of data points marked as "illicit" in the Elliptic Bitcoin Dataset to be used. The sum of `<n_illicit> + <n_licit>` determines the dataset size. This should be an integer.
- `<data_seed>` is the seed for the random sampling used to obtain the `<n_illicit> + <n_licit>` data points from the dataset. This should be an integer.
- `<data_file>` is the CSV file containing the preprocessed dataset. This should be `"elliptic_preproc.csv"`.

**NOTE**: The above command will use a single process. In order to take advantage of parallelisation you should run the Python script via the appropriate MPI command (e.g. via `mpirun`, `mpiexec`, `srun`, etc.).
