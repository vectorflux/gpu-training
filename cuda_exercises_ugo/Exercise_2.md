Exercise 2_0: sum vectors, fixed number of threads

Author: Ugo Varetto

Goal: compute the sum of two 1D vectors using a number of threads greater than or equal to the number of vector elements and not evenly divisible by the block size

Rationale: shows how to implement a kernel with a computation/memory configuration that matches the domain layout. Each threads computes at most one element of the output vector. Compute the scalar product of two 1D vectors using a number of threads lower than the size of the output vector

Solution: The code needs to be extended to support the following functionality.

    number of elements in the output array = E
    number of threads per block = Tb 

The number of blocks needs to be ( E + Tb - 1 ) div Tb where 'div' is the integer division operator. Each thread on the GPU computes one(thread id < vector size) or zero( thread id >= vector size) elements of the output vector.


Workflow:

1) compute launch grid configuration 2) allocate data on host (cpu) and device (gpu) 3) copy data from host to device 4) launch kernel 5) read data back 6) consume data (in this case print result) 7) free memory

Compilation: nvcc -arch=sm_13 2_0_sum-vectors.cu -o 2_0_sum-vectors

Execution: ./2_0_sum-vectors

Notes:

    The first vector is set to all ones, the second vector to all twos. 

    the code is C++ also because the default compilation mode for CUDA is C++, all functions are named with C++ convention and the syntax is checked by default against C++ grammar rules 

    -arch=sm_13 allows the code to run on every card available on Eiger and possibly even on students' laptops; it's the identifier for the architecture before Fermi (sm_20) 

    -arch=sm_13 is the lowest architecture version that supports double precision 

    the example can be extended to read configuration data and array size from the command line and could be timed to investigate how performance is dependent on single/double precision and thread block size 

Code: 2_0_sum-vectors.cu

Assignment:

    Compile and run the program. Looking at the result vector "vout", it becomes clear that the code contains bugs. Carefully go through the code to determine the error, and correct it. Hint: it is one or more cuda calls. 

    The code should run now an produce the correct result vector (all 3's). Now change the vector length by adding one (i.e., 65537). Retry the code to see what happens. 

    Alter the code to implement the blocking scheme defined in the Solution. Alter the kernel invocation (or the kernel itself) to reflect the change. Rerun to the code to make sure it produces correct answers. 

    Take a look at the alternative "sum_vectors" kernel implementation 2_1_sum-vectors.cu. This version will go through all entries local to the thread, even if the number is not constant in all blocks. This version "does not comple". Change the line with the "ASSIGNMENT" comment appropriate. Ascertain that this version works, with the existing definition of "NUMBER_OF_BLOCKS" for arbitrary "VECTOR_SIZE". 

Compilation: nvcc -arch=sm_13 2_1_sum-vectors.cu -o 2_1_sum-vectors 
