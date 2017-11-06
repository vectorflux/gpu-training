 Exercise 4: Parallel Dot Product

Author: Ugo Varetto

Goal: compute the dot product of two vectors

Rationale: shows how to perform the dot product of two vectors as a parallel reduction with the last reduction step either performed on the CPU (Ex. 4) or on the GPU (Ex 4_1). In the latter case, the last step must be done through synchronized access to a shared variable. This feature is not supported and does not run on architecture < 2.0 (Fermi); return value stays set to zero.


Solution: store scalar products in local cache and iterate over cache elements performing incremental sums

Workflow: see documentation in individual files

Compilation:

    nvcc -arch=sm_13 4_parallel-dot-product.cu -o 4_parallel-dot-product 

    nvcc -arch=sm_20 4_1_parallel-dot-product.cu -o 4_1_parallel-dot-product 

Execution: ./4_parallel-dot-product or ./4_1_parallel-dot-product

Notes:

    shared memory "must" be allocated at compile time, on the same OpenCL allows client code to specify size of shared memory at kernel invocation time 

    the code is C++ also because the default compilation mode for CUDA is C++, all functions are named with C++ convention and the syntax is checked by default against C++ grammar rules 

    -arch=sm_13 allows the code to run on every card available on Eiger and possibly even on students' laptops; it's the identifier for the architecture before Fermi (sm_20) 

    -arch=sm_13 is the lowest architecture version that supports double precision 

    the example can be extended to read configuration data and vector size from the command line 

Code:

    4_parallel-dot-product.cu -- final reduction on CPU 

    4_1_parallel-dot-product-atomics.cu -- final reduction on GPU 

Assignment:

    Take a look at "4_parallel-dot-product.cu", concentrating on the iterative halving of the block size. This is a working example. Test it out to ascertain that results on CPU (reference version) and GPU are the same. 

    Turn your attention to "4_1_parallel-dot-product-atomics.cu". Try compiling and running the code on any GPU. The GPU version does not give the same results as the CPU version, because the addition operation is not atomic. Can you explain the results on the GPU? 

    Correct the code by performing an : "Atomic Add" operation. Try compiling your new version with: 

  nvcc -arch=sm_13  4_1_parallel-dot-product.cu -o 4_1_parallel-dot-product

    Try compiling and running the code on GTX285 (Geforce) or C1070 (Tesla) cards with: 

 nvcc -arch=sm_20  4_1_parallel-dot-product.cu -o 4_1_parallel-dot-product
 ./4_1_parallel-dot-product

Why does this fail?

    Finally try compiling and running the code on M2050 or C2070 cards with arch=sm_20. If you have programmed the atomic add correctly, this should now give the correct result. 
