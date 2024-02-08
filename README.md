# qml-cutensornet

Code for our experiments on Quantum Kernel methods being simulated on NVIDIA GPUs via pytket-cutensornet.

The Elliptic Bitcoin Dataset needs to be downloaded and stored in the folder ``elliptic_bitcoin_dataset``. The two files should be ``elliptic_txs_classes.csv`` and ``elliptic_txs_features.csv``.

### Installation

Apart from Python and pytket, this branch makes use of Julia, ITensors.jl, pyCall and a local package KernelPkg.jl.

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

 ```

Finally, to install *KernelPkg.jl*, move the contents of the folder `KernelPkg` in this repository to `~/.julia/dev/KernelPkg`. Then, open the Julia REPL and press `]` to enter the package manager. Locally install the package by typing:
`dev ~/.julia/dev/KernelPkg`.
