
ALENEX2022 supplementary materials for 
accompanying paper: Approximating 1-Wasserstein Distance between Persistence Diagrams by Graph Sparsification
                
by Tamal Dey and Simon Zhang
## 1-Wasserstein Distance
Subdirectory of our implementation to compute a ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%201&plus;O%28%5Cepsilon%29) approximation of the 1-Wasserstein Distance  


### Requirements:

A Unix-based system such as Ubuntu 20.04; Should not compile on a Windows system.

CMake version >=3.1

CUDA version >=10.1

compiler with c++11 support:

GCC version >= 7.3.0

The GPU requirement is not that heavy, it just needs >=s^2*n memory for the min-cost flow graph where s is the input sparsity parameter and n is the number of nodes in the min-cost flow graph. 

However for high performance, as shown in the experiments of the paper, you should have multiple CPU cores. For example, the experiments in the paper were run with 48 cores to speedup the WSPD construction stage. 

More parallelism can be exploited in the WSPD construction stage for a polylog(s^2n) depth algorithm as in [5]. Our implementation is a simplification of such an algorithm. 
### Building:

```
git clone --recurse-submodules https://github.com/simonzhang00/pdoptflow.git
cd pdoptflow/pdoptflow
mkdir build 
cd build
cmake -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) ..
make CXX=$(which g++) CC=$(which gcc)
```
This will auto build the lemon-1.3.1 library [1] using ```cmake .. && make```. Be sure you are running a system with ```cmake``` and ```make``` installed. 

After building the lemon library the command will build the executables ```1-wasserstein_dist```,```k-experiments``` and testing executables in the testing_dir/ directory under the build folder.

To run the standard battery of 800+ tests on GPU, type: ```make test``` 

The 879_test_battery executable may take some time to compute on a laptop (a 6GB device memory GPU should be able to run the tests as they were designed)

If you want to see the s values and approximately computed distances for 879_test_battery, cd to testing_dir and run the command:

```
./879_test_battery ../../../datasets/tests/wasserstein_tests_QA_exact.txt 
``` 
similarly for more_tests, in the testing_dir/ directory of the build folder, run:

```
./more_tests ../../../datasets/more_tests/wasserstein_tests2_QA_exact.txt 
``` 
### Usage:

There will be an executable and a .so file built by `make`. The .so file can be copied to the python directory to be used as a python module.

```
1-wasserstein_dist [--options] file1 file2  
```

```
Options:

--s <s> s is parameter for WSPD and determiner of delta (see paper for formulas)
--additive <b> b is additive coefficient for early stopping
--multiplicative <C> C is multiplicative coefficient for early stopping
    b+sqrt(mn)*C is the stopping number of blocks to be found by block search algorithm
```
if --s < s > is not specified, then s is default to 5.

defaults of b and C are determined by tuning.

*WARNING* If you enter a very large s such as 1000 for a pair of large datasets, you will very likely crash your system or you will get an out of memory error especially if you are just using a laptop. This is because by providing a large s value, you are inducing a graph that is order O(n^2) number of arcs for network simplex implementation. A good rule of thumb is that s^2*(total number of persistence points)*8 bytes should be much smaller than DRAM. So for example, if there are 200000 persistence points and only 64 GB DRAM, s << 200. s<=18 is effective for most systems.

*SUGGESTION* In fact, PDoptFlow works best for s<=18 (a guaranteed 2.3 approximation). There are no theoretical guarantees for s<2; however this is actually a superb range of s values for performance. The empirical error is usually still very low (see the paper).

The output of PDoptFlow is an approximation of the true 1-Wasserstein distance between persistence diagrams. 

Let d_exact be the true distance and d_approx be the output, then one can check the relative error as 
    
    | d_exact - d_approx | / d_exact
    
As shown in the experiments in the paper, the relative error is usually quite low even for s<=32 although the theoretically derived s from the paper is usually much higher. The theoretical bound assures that for s sufficiently large, a low relative error will be achieved.

file1 and file2 must contain persistence diagrams in hera [1] format, which is in ascii text format (usually cleaned and parsed from the output of some persistent homology software) 

It is formatted as one point per line, empty lines are ignored, comments can be made with #:
```
# this is the hera persistence diagram ascii file input format:
x_1 y_1 # two real numbers per line
...
# empty lines or comments are ignored
x_n y_n 
```
To run some sample datasets:

The following compares the Athens and Beijing datasets with the 1-Wasserstein distance with theoretical relative error <=0.5.

```./wasserstein_dist ../../datasets/PD_lower_star_Athens_760x585-csv.txt ../../datasets/PD_lower_star_Beijing_1654x2270-csv.txt --s 40 ```

#### Verbosity:

To turn on all counting and profiling print outs, go to the include/config.h file and uncomment the ```#define VERBOSE``` line.

To turn on profiling for network simplex, go to 1-wasserstein/deps/lemon-1.3.1-par/lemon/network_simplex.h and modify the #define QUIET there. 

#### Stopping Criterion:

The stopping criterion is: 

if number of blocks> ADDITIVE_COEFF+MULTIPLICATIVE_COEFF*sqrt(m*n) then stop early.

** If you are not satisfied with an early abort, then simply rerun with the same s or try s+1 or s+2 and increase ADDITIVE_COEFF ; since the algorithm is randomized, the transshipment network will be slightly different each time, within theoretical bounds. **

### Parallelization of the block pivot search

We have implemented a parallel block pivot search algorithm on multicore in deps/lemon-1.3.1-par/lemon/network_simplex.h, see enum PivotRule and the element PARALLEL_BLOCK_SEARCH. To enable parallelization at compile time, uncomment the #define PARALLEL in deps/lemon-1.3.1-par/lemon/network_simplex.h. This is commented out by default so that OpenMP 4.5 is not required by default. High versions of OpenMP are often not available on many machines. There should not be much of a performance difference between the parallel block pivot search and the sequential block pivot search since large block sizes are not necessarily better than smaller block sizes.



## References:
 
1. Dezső, Balázs, Alpár Jüttner, and Péter Kovács. "LEMON–an open source C++ graph template library." Electronic Notes in Theoretical Computer Science 264.5 (2011): 23-45.
2. Flamary, Rémi, and Nicolas Courty. "Pot python optimal transport library." GitHub: https://github.com/rflamary/POT (2017).
3. Kerber, Michael, Dmitriy Morozov, and Arnur Nigmetov. "Geometry helps to compare persistence diagrams." Journal of Experimental Algorithmics (JEA) 22 (2017): 1-20.
4. Maria, Clément, et al. "The Gudhi library: Simplicial complexes and persistent homology." International Congress on Mathematical Software. Springer, Berlin, Heidelberg, 2014.
5. Yiqiu  Wang,  Shangdi  Yu,  Yan  Gu,  and  Julian  Shun.Fast parallel algorithms for euclidean minimum spanning  tree  and  hierarchical  spatial  clustering.   In Proceedings of the 2021 International Conference on Management of Data, pages 1982–1995, 2021