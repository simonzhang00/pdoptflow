# PDoptFlow
This is the supplementary material for the ALENEX2022 paper:

Approximating 1-Wasserstein Distance between Persistence Diagrams by Graph Sparsification
by Tamal Dey and Simon Zhang

Copyright © 2020, 2021 Tamal Dey and Simon Zhang

Code Author: Simon Zhang

The implementation involves some algorithmic simplifications. The repository can reproduce the paper's results.

PDoptFlow is licensed under the GNU General Public License

To download then install (Unix-based only):

See pdoptflow/README.md for system requirements.
```
git clone --recurse-submodules https://github.com/simonzhang00/pdoptflow.git
cd pdoptflow
pip3 install .
```

or to directly install:
```
pip3 install git+https://github.com/simonzhang00/pdoptflow.git
```

For manual python installation, see python/ directory.

see subdirectories of
1. datasets/ for the datasets and automated test cases
2. pdoptflow/ for computing the 1-wasserstein distance
3. python_experiments/ for all scripts to reproduce the experiments in the paper

suggestion: checkout the README.md in python_experiments/ first

### Usage:
```
from pdoptflow import W1
import numpy as np
numpyA= np.random.random((100,2))
numpyB= np.random.random((100,2))
s= 3
additive_pivots_till_abort= 10000
approx= W1.wasserstein1(numpyA, numpyB, s, additive_pivots_till_abort)
```

