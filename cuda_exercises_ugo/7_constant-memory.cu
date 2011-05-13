//WORK IN PROGRESS !!!!!

// #CSCS CUDA Training 
//
// #Exercise 7 - sum vectors, fix number of threads per block, overlap communication and computation
//
// #Author Ugo Varetto
//
// #Goal: compute the sum of two vectors overlapping communication and computation
//
// #Rationale: using streams it is possible to subdivide computation and memory transfer
//             operations in separate execution queue which can be executed in parallel;
//             specifically it is possible to execute kernel computation while concurrently
//             transferring data between host and device
//
// #Solution: subdivide domain into chunks, iterate over the array and at each
//            iteration issue asynchronous calls to memcpy and kernels
//            (always asynchronous) in separate streams. Note that
//            on the GPU operations are split into one queue per operation type
//            specifically: all copy operations from different streams go into the same
//            queue in the same order as specified by client code, likewise all
//            kernel invocations go into the same 'kernel invocation queue'.
//            Now: if any copy operation (regardless of the associated stream) depends
//            on e.g. a kernel execution then all subsequent copy operations in all streams must wait
//            for the dependent copy operation to complete before they are executed; this means
//            that client code is responsible for properly queueing operations to avoid conflicts. 
//
// #Code: flow:
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host ro device
//        4) create streams
//        5) iterate over array elements performing memory transfers and
//           kernel invocation: at each iteration the number of elements
//           being processed by separate streams is   VECTOR_CHUNK_SIZE x NUMBER_OF_STREAMS
//        6) synchronize streams to wait for end of execution 
//        7) consume data (in this case print result)
//        8) free memory, streams and events (used to time operations)
//        
// #Compilation: [optimized] nvcc -arch=sm_13 2_2_sum-vectors-overlap.cu -o sum_vectors-overlap
//               [wrong ordering, no overlap] nvcc -DSTREAM_NO_OVERLAP -arch=sm_13 2_2_sum-vectors-overlap.cu -o sum_vectors-overlap
//                 
//
// #Execution: ./sum-vectors-overlap
//             note that you might experience some hysteresis! on Win 7 64bit, CUDA RC2 at each compilation
//             it takes a few runs before the new ordering scheme becomes active!!! 
//
// #Note: page locked memory required for async/stream operations
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card available on Eiger and possibly even
//        on students' laptops; it's the identifier for the architecture before Fermi (sm_20)
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
//
// #Note: the example can be extended to read configuration data and array size from the command line and
//        investigating the optimal configuration for number of streams and chunk size


#include <cuda.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>


typedef float real_t;


static const int NUM_WEIGHTS = 256;
static const int NUMBER_OF_BLOCKS  = 2048;
static const int THREADS_PER_BLOCK = NUM_WEIGHTS;
static const int HALF_WARP = 16;

__constant__ real_t weights[ NUM_WEIGHTS ];


// out[ global thread id ] = in[ global thread id ] * weights[ block id ];
// each thread in the group accesses the same element in the constant weight array,
// each read from 16(half warp) adjacent threads result in a single transfer operation
__global__ void weight_mul_broadcast( const real_t* vin, real_t* out, int num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do note write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = vin[ xIndex ] * weights[ threadIdx.x / HALF_WARP ];
}

// out[ global thread id ] = in[ global thread id ] * weights[ local thread id ];
// each thread in the group accesses a different weight: each access from half warp
// threads is serialized i.e. it will take 16 separate read operations to fill a group of 16 output
// elements as comparaed to 16 parallel transfers or less(when coalesced) in the case of global
// memory 
__global__ void weight_mul_serial( const real_t* vin, real_t* out, int num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do note write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = vin[ xIndex ] * weights[ threadIdx.x ];
}


// out[ global thread id ] = in[ global thread id ] * weights[ local thread id ];
// each thread in the group accesses a different weight: each access from half warp
// threads is serialized i.e. it will take 16 separate read operations to fill a group of 16 output
// elements as comparaed to 16 parallel transfers or less(when coalesced) in the case of global
// memory 
__global__ void weight_mul_global( const real_t* vin, const real_t* weights, real_t* out, int num_elements ) {
    // compute current thread id
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;          
    // since we assume that num threads >= num element we need to make sure we do note write outside the
    // range of the output buffer 
    if( xIndex < num_elements ) out[ xIndex ] = vin[ xIndex ] * weights[ threadIdx.x ];
}


// generate constant element
    struct GenSeq {
        static int v_; 
        GenSeq( real_t v )  { v_ = v; }
        real_t operator()() const { return ++v_; }
    };

    int GenSeq::v_ = 0;

//------------------------------------------------------------------------------
int main( int , char**  ) {
      
    const int VECTOR_SIZE = NUMBER_OF_BLOCKS * THREADS_PER_BLOCK;
    const int BYTE_SIZE = sizeof( real_t ) * VECTOR_SIZE;
      
    // host allocated storage; page locked memory required for async/stream operations
    std::vector< real_t > v( VECTOR_SIZE, 1.f );
    std::vector< real_t > host_w( NUM_WEIGHTS );
    std::vector< real_t > vout( VECTOR_SIZE );
   
    std::generate( v.begin(), v.end(), GenSeq( 0.0f ) );

    cudaMemcpyToSymbol( weights, &host_w[ 0 ], sizeof( real_t ) * NUM_WEIGHTS );

    // gpu allocated storage: number of arrays == number of streams == 2
    real_t* dev_vin  = 0;
    real_t* dev_vout = 0;
    real_t* dev_w    = 0;

    cudaMalloc( &dev_vin,  BYTE_SIZE );
    cudaMalloc( &dev_vout, BYTE_SIZE );
    cudaMalloc( &dev_w, sizeof( real_t) * NUM_WEIGHTS );
    
    // events; for timing
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop  = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );
    weight_mul_broadcast<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_vout, VECTOR_SIZE );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float e = float();
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Broadcast:  " << e << " ms" << std::endl;

    cudaEventRecord( start, 0 );
    weight_mul_serial<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_vout, VECTOR_SIZE );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Serialized: " << e << " ms" << std::endl;

    cudaEventRecord( start, 0 );
    weight_mul_global<<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( dev_vin, dev_w, dev_vout, VECTOR_SIZE );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    std::cout << "Global:     " << e << " ms" << std::endl;

    // free memory
    cudaFree( dev_vin  );
    cudaFree( dev_w    );
    cudaFree( dev_vout );

    // release events
    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

    return 0;
}
