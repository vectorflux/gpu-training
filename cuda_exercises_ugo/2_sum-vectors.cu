// #CSCS CUDA Training 
//
// #Exercise 2 - sum vectors
//
// #Author Ugo Varetto
//
// #Goal: compute the scalar product of two 1D vectors using a number of threads lower than the number of elements
//        and not evenly divisible by the block size (i.e. num threads per block);
//        number of threads per block must be > 1       
//
// #Rationale: shows how to implement a kernel with a computation/memory configuration independent on the 
//             domain data; this is required in case the data is bigger than the compuation grid (see exercise 1)
//
// #Solution: 
//          (1) 
//          . total number of threads = T
//          . number of threads per block = Tb          
//          The number of blocks is = ( T + Tb - 1 ) div Tb where 'div' is the integer division operator           
//          (2)
//          Since the total number of threads must be assumed smaller than the vector size, each
//          GPU thread must iterate over more than one array element
//
// #Code: typical flow:
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host ro device
//        4) launch kernel
//        5) read data back
//        6) consume data (in this case print result)
//        7) free memory
//        
// #Compilation: nvcc -arch=sm_13 2_sum_vectors.cu -o sum_vectors
//
// #Execution: ./sum_vectors 
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card available on Eiger and possibly even
//        on students' laptops; it's the identifier for the architecture before Fermi (sm_20)
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
//
// #Note: the example can be extended to read configuration data and array size from the command line
//        and could be timed to investigate how performance is dependent on single/double precision
//        and thread block size


#include <cuda.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

typedef float real_t;

// CUDA kernel invoked by CPU (host) code; return type must always be void
__global__ void sum_vectors( const real_t* v1, const real_t* v2, real_t* out, size_t num_elements ) {
    // compute current thread id
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // iterate over vector: grid can be smaller than vector, it is therefore
    // required that each thread iterate over more than one vector element
    while( xIndex < num_elements ) {
        out[ xIndex ] = v1[ xIndex ] + v2[ xIndex ];
        xIndex += gridDim.x * blockDim.x;
    }
}


//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int VECTOR_SIZE = 0x10000 + 1; //vector size 65537
    const int NUMBER_OF_THREADS = VECTOR_SIZE / 4; //number elements processed in parallel
    const int SIZE = sizeof( real_t ) * VECTOR_SIZE; // total size in bytes
    const int THREADS_PER_BLOCK = 32; //number of gpu threads per block
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int BLOCK_SIZE = ( NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    // if number of threads is not evenly divisable by the number of threads per block 
    // we need an additional block; the above code can be rewritten as
    // if( NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK;
    // else BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1 

    // host allocated storage; use std vectors to simplify memory management
    // and initialization
    std::vector< real_t > v1  ( VECTOR_SIZE, 1.f ); //initialize all elements to 1
    std::vector< real_t > v2  ( VECTOR_SIZE, 2.f ); //initialize all elements to 2   
    std::vector< real_t > vout( VECTOR_SIZE, 0.f ); //initialize all elements to 0

    // gpu allocated storage
    real_t* dev_in1 = 0; //vector 1
    real_t* dev_in2 = 0; //vector 2
    real_t* dev_out = 0; //result value
    cudaMalloc( &dev_in1, SIZE );
    cudaMalloc( &dev_in2, SIZE );
    cudaMalloc( &dev_out, SIZE  );
    
    // copy data to GPU
    cudaMemcpy( dev_in1, &v1[ 0 ], SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_in2, &v2[ 0 ], SIZE, cudaMemcpyHostToDevice );

    // execute kernel
    sum_vectors<<<BLOCK_SIZE, THREADS_PER_BLOCK>>>( dev_in1, dev_in2, dev_out, VECTOR_SIZE );
    
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    
    // print first and last element of vector
    std::cout << "result: " << vout[ 0 ] << ".." << vout.back() << std::endl;

    // free memory
    cudaFree( dev_in1 );
    cudaFree( dev_in2 );
    cudaFree( dev_out );

    return 0;
}
