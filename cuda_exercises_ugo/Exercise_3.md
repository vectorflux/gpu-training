Exercise 3: Matrix Transpose

Author: Ugo Varetto

Goal: compute the transpose of a matrix and analyze performance with on-board counters

Rationale: shows how to perform operations on a 2D grid and how to use the GPU for data initialization

Solution:

3.0) simply compute the thread id associated with the element and copy transposed data into output matrix (no timing)

3.1) compute the thread id associated with the element and copy the transposed data into the output matrix; wrap kernel calls with event recording and print time information

3.2) optimization: copy input matrix elements into shared memory blocks and write transposed elements reading from shared memory (which can be an order of magnitude faster). Access is coalesced if the block size is a multiple of a half warp i.e. 16

Workflow: see documentation in individual files

Compilation: nvcc -arch=sm_13 xxx.cu -o xxx

where xxx is one of: 3_0_transpose, 3_1_transpose-timing, 3_2_transpose-timing-coalesced

Execution: ./xxx

Notes:

    the code is C++ also because the default compilation mode for CUDA is C++, all functions are named with C++ convention and the syntax is checked by default against C++ grammar rules 

    -arch=sm_13 allows the code to run on every card available on Eiger and possibly even on students' laptops; it's the identifier for the architecture before Fermi (sm_20) 

    -arch=sm_13 is the lowest architecture version that supports double precision 

    the example can be extended to read configuration data and matrix size from the command line 

Code:

    3_0_transpose.cu 

    3_1_transpose-timing.cu -- Timing additions 

    3_2_transpose-timing-coalesced.cu 

    3_0_transpose-sol.cu -- Solution to 3_0 

    3_2_transpose-timing-coalesced-sol.cu -- Solution to 3_2 

Assignment:

    Start with 3_0. The code is not complete, and the output matrix contains gibberish. In the "transpose" kernel, complete the calculation of "row", "col" and "input_index" in order to obtain the correct result. 

    In 3_0: swap the new definitions of "row" and "col" in "transpose". Does this change the output matrix? Why or why not? 

    Version 3_1 adds the timers (and contains the solutions to the previous exercises. Benchmark 3_1 on all GPU types. Does compiler optimization "-O3" change the timings considerably? 

    Version 3_2 has the infrastructure for the shared "tile", but it is not yet used. Add the code needed: set the appropriate values into "tile", revise the output index if needed, perform synchronization if needed, and assign "out" the appropriate values from "tile". 

    Benchmark 3_2 on all the GPU types. Does the improvement ratio 3_1/3_2 depend on the GPU type? How about matrix size? 
