Exercise 5: Matrix-Matrix Multiplication

Author: Ugo Varetto

Goal: compute the product of two matrices

Rationale: Even if matrix-matrix multiplication is conceptually simple, a high performance implementation is quite difficult. Here we propose two different kernels, "matmul" and "block_matmul" to illustrate how performance enhancement can be undertaken.

Solution: store scalar products in local cache and iterate over cache elements performing incremental sums

Workflow: see documentation in individual files

Compilation:

    [standard] nvcc -arch=sm_13 5_matmul.cu -o 5_matmul
    [block multiply] nvcc -DBLOCK_MULTIPLY -arch=sm_13 5_matmul.cu -o 5_block_matmul 

Execution: ./5_matmul or ./5_block_matmul

Notes:

    try on both G90 and GF100 architectures to verify the impact of L1 cache 

Code: 5_matmul.cu

Assignment:

    The current code "does not compile". Go to the "get_matrix_element" and insert the correct values in place of the question marks. If done properly you should be able to compile and run the default (i.e. non-blocked) matmul version. 

    Compile with -DCOMPARE_RESULTS which compares the results of GPU and CPU versions. The test fails. Now change the "typedef float real_t" to "typedef double real_t". The test now passes. Explain this. Find out for what value of "EPSILON" floats and doubles just barely pass the comparison test for the default size. Does this vary depending on the GPU? 

    Reread the lecture material on the matrix multiply, and determine the subtle differences between the versions presented and this version. Then visit the kernel "block_matmul" and fill in the question marks in the lines with an "ASSIGNMENT" comment. 

    Timers have already been placed around the two matrix multiplication kernels. Benchmark both kernels on all GPU types, for several different "BLOCK_SIZE" values (currently 64, which yields a problem size of 64 times 16 threads per block = 1024). 

    To make a fairer comparison, place timers around the device-to-host "cudaMemcpy" of the result matrix "dev_mout". Even if this data transfer time is included, the timings are still not representative for the general case. Why not? 

    Extra exercise: In this example code, shared memory "tiles" M1 and M2 are employed similar to exercise 3. These tiles are allocated statically. From CUDA SDK 3.2 onward it is also possible to allocate device shared memory dynamically. First one defines a cache: 

 extern __shared__ real_t cache[];

In this implementation, the tile is more generally defined as TILE_ROWS by TILE_COLUMNS, with

  const int TILE_COLUMNS = blockDim.x;
  const int TILE_ROWS    = blockDim.y;

Since both M1 and M2 have to be accommodated in the cache, they have to be aliased with the appropriate offsets:

  real_t* M1 = &cache[ 0 ];
  real_t* M2 = &cache[ TILE_COLUMNS * TILE_ROWS];     

The cached shared memory size is specified as an additional argument passed to the kernel. I.e., instead of

 block_matmul<<<BLOCKS, THREADS_PER_BLOCK>>> ...

we need:

 const size_t SHARED_MEMORY_SIZE=2*THREADS_PER_BLOCK.x*THREADS_PER_BLOCK.y*sizeof( real_t);  

 block_matmul<<<BLOCKS, THREADS_PER_BLOCK, SHARED_MEMORY_SIZE >>> ...

Finally, accessing into M1 and M2 has to be converted to a 1-dimensional index, i.e., instead of Implement dynamically allocated shared memory for M1 and M

 M1[ row ][ col ] = get_matrix_element( m1, b, blockRow, col, row, m1_columns );

we need:

 M1[ row * TILE_COLUMNS + col ] = get_matrix_element( m1, b, blockRow, col, row, m1_columns );

Implement dynamically allocated shared memory for M1 and M2. Benchmark the code to see if this has any performance implications.

Complete codes:

5_matmul-sol.cu -- working version (without dynamic shared memory)

5_matmul-dynamic-shared-mem.cu -- working version with dynamic shared memory 
