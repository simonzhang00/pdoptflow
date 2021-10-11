import gudhi.hera
import numpy as np
import time
import sys

if __name__ == "__main__":
    tic = time.time()
    filename1 = sys.argv[1]
    with open(filename1, 'r') as fobj:
        X = np.array([[float(num) for num in line.split()] for line in fobj])
    print(X.shape)

    filename2 = sys.argv[2]
    with open(filename2, 'r') as fobj:
        Y = np.array([[float(num) for num in line.split()] for line in fobj])
    print(Y.shape)

    dist= gudhi.hera.wasserstein_distance(X,Y, order =1, internal_p=2)
    print(dist)
    toc = time.time()
    print("total hera computation time: ", toc-tic)