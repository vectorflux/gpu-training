// WORK IN PROGRESS !!
// #CSCS CUDA Training 
//
// #Exercise 2_2 - sum vectors, fix number of threads per block, overlap communication and computation
//
// #Author Ugo Varetto
//
// #Goal: compute the scalar product of two 1D vectors using a number of threads greater than or equal to
//        the number of vector elements and not evenly divisible by the block size 

// #Rationale: shows how to implement a kernel with a computation/memory configuration that matches
//             the domain layout. Each threads computes at most one element of the output vector.
//
// #Solution: 
//          . number of elements in the output array = E
//          . number of threads per block = Tb          
//          The number of blocks is = ( E + Tb - 1 ) div Tb where 'div' is the integer division operator   
//          Each thread on the GPU computes one(thread id < vector size) or zero( thread id >= vector size)
//          elements of the output vector.        
//
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
// #Compilation: nvcc -arch=sm_13 2_0_sum_vectors.cu -o sum_vectors_1
//
// #Execution: ./sum_vectors_1 
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

// In this case the kernel assumes that the computation was started with enough threads to cover the entire domain.
// This is the preferred solution provided there are enough threads to cover the entire domain which might not be the
// case in case of a 1D grid layout (max number of threads = 512 threads per block x 65536  blocks = 2^25 = 32 Mi threads)
__global__ void sum_vectors( const real_t* v1, const real_t* v2, real_t* out, size_t num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do note write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = v1[ xIndex ] + v2[ xIndex ];
}




//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int VECTOR_SIZE = 0x10000; //vector size 65536
    const int NUMBER_OF_CHUNKS = 4;
    const int VECTOR_CHUNK_SIZE = VECTOR_SIZE / NUMBER_OF_CHUNKS;
    const int FULL_SIZE = sizeof( real_t ) * VECTOR_SIZE;
    const int CHUNK_SIZE = FULL_SIZE / NUMBER_OF_CHUNKS; // total size in bytes
    const int THREADS_PER_BLOCK = 32; //number of gpu threads per block
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int NUMBER_OF_BLOCKS = ( VECTOR_SIZE + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    // if number of threads is not evenly divisable by the number of threads per block 
    // we need an additional block; the above code can be rewritten as
    // if( NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK;
    // else BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1 

    // host allocated storage; page locked memory required! 
    std::vector< real_t > init  ( VECTOR_SIZE, 1.f ); //initialize all elements to 1
    real_t* v1   = 0;
    real_t* v2   = 0;
    real_t* vout = 0;
     
    cudaHostAlloc( &v1, FULL_SIZE, cudaHostAllocDefault );
    cudaHostAlloc( &v2, FULL_SIZE, cudaHostAllocDefault );
    cudaHostAlloc( &vout, FULL_SIZE, cudaHostAllocDefault );  

    std::copy( init.begin(), init.end(), v1 );
    std::copy( init.begin(), init.end(), v2 );

    // gpu allocated storage
    real_t* dev_in10 = 0;
    real_t* dev_in11 = 0;
    real_t* dev_in20 = 0;
    real_t* dev_in21 = 0;
    real_t* dev_out0 = 0;
    real_t* dev_out1 = 0;

    cudaMalloc( &dev_in10, CHUNK_SIZE );
    cudaMalloc( &dev_in11, CHUNK_SIZE );
    cudaMalloc( &dev_in20, CHUNK_SIZE );
    cudaMalloc( &dev_in21, CHUNK_SIZE );
    cudaMalloc( &dev_out0, CHUNK_SIZE );
    cudaMalloc( &dev_out1, CHUNK_SIZE );
    
    // streams
    cudaStream_t stream0 = cudaStream_t();
    cudaStream_t stream1 = cudaStream_t();
    cudaStreamCreate( &stream0 );
    cudaStreamCreate( &stream1 );

    // events
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop  = cudaEvent_t();
    cudaEventCreate( &start );



    // computation (wrong order)
    for( int i = 0; i < VECTOR_SIZE; i += 2 * VECTOR_CHUNK_SIZE ) {
        cudaMemcpyAsync( dev_in10, v1 + i, SIZE, cudaMemcpyHostToDevice, stream0 );
        cudaMemcpyAsync( dev_in20, v2 + i, SIZE, cudaMemcpyHostToDevice, stream0 );
        sum_vectors<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream0 >>>( dev_in10, dev_in20, dev_out0, VECTOR_SIZE );
        cudaMemcpyAsync( vout + i, dev_out0, SIZE, cudaMemcpyDeviceToHost, stream0 );

        cudaMemcpyAsync( dev_in11, v1 + i + CHUNK_SIZE, SIZE, cudaMemcpyHostToDevice, stream1 );
        cudaMemcpyAsync( dev_in21, v2 + i + CHUNK_SIZE, SIZE, cudaMemcpyHostToDevice, stream1 );
        sum_vectors<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream1 >>>( dev_in11, dev_in21, dev_out1, VECTOR_SIZE );
        cudaMemcpyAsync( vout + i + N, dev_out1, SIZE, cudaMemcpyDeviceToHost, stream1 );
    }
    cudaStreamSynchronize( stream0 );
    cudaStreamSynchronize( stream1 );
    
    // print first and last element of vector
    std::cout << "result: " << vout[ 0 ] << ".." << vout[ VECTOR_SIZE - 1 ] << std::endl;

    // free memory
    cudaFree( dev_in10 );
    cudaFree( dev_in11 );
    cudaFree( dev_in20 );
    cudaFree( dev_in21 );
    cudaFree( dev_out0 );
    cudaFree( dev_out1 );
    cudaFreeHost( v1 );
    cudaFreeHost( v2 );
    cudaFreeHost( vout );

    return 0;
}
