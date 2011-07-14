// "mpi + cuda reduction" 

#ifdef GPU
#include <cuda.h>
#endif
#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpierr.h"
#include <cmath>
#include <algorithm>
#include <sstream>

// compilation with mvapich2:
// nvcc -L/apps/eiger/mvapich2/1.6/mvapich2-gnu/lib -I/apps/eiger/mvapich2/1.6/mvapich2-gnu/include \
// -libumad -lmpich -lpthread -lrdmacm -libverbs -arch=sm_20 -DGPU \
// ~/projects/gpu-training/trunk/cuda_exercises_ugo/resources/mpiscratch/mpicuda2.cu


// note: using single precision floats because that's the only supported type
//       for atomics on CUDA 4

// note: experiment with different number of MPI tasks per GPU/node; using
//       256 Mi floats, 16 MPI tasks on two nodes (8 per node, 4 per GPUs)
//       CUDA fails to allocate memory exaclty for one task on each node;
//       Everything works fine with the same data with 8 tasks (4 per node, 2 per GPU ).

typedef float real_t;
#define MPI_REAL_T_ MPI_FLOAT

//------------------------------------------------------------------------------
#ifdef GPU
const int BLOCK_SIZE = 128;

__global__ void dot_product_kernel( const real_t* v1, const real_t* v2, int N, real_t* out ) {
    __shared__ real_t cache[ BLOCK_SIZE ];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= N ) return;
    cache[ threadIdx.x ] = 0.f;
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }    
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx.x + i ];
        __syncthreads();
        i /= 2; //not sure bitwise operations are actually faster
    }
    if( threadIdx.x == 0 ) atomicAdd( out, cache[ 0 ] );   
}
#endif


//------------------------------------------------------------------------------
int main( int argc, char** argv ) {

    int numtasks = 0;
    int task     = 0;
    // INIT ENV
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &task  ) );
    std::vector< char > nodeid( MPI_MAX_PROCESSOR_NAME, '\0' );
    int len = 0;
    MPI_( MPI_Get_processor_name( &nodeid[ 0 ], &len ) );
    
    // RETRIEVE TOTAL NUMBER OF NODES USED, is there an easier way ?
    MPI_( MPI_Isend( &nodeid[ 0 ]...;
    if( task == 0 ) {
        typedef std::map< std::string, int > NodeCount;
        NodeCount ncount;
        std::vector< char > n( MPI_MAX_PROCESSOR_NAME, '\0' );
        for( int r = 0; r != numtasks; ++r ) {
            MPI_( MPI_Recv( &n[ 0 ],...;
            ++ncount[ &n[ 0 ] ];    
        }
        for( int r = 0; r != numtasks; ++r ) {
            MPI_( MPI_Isend( &nodecount...;
        }
    }
    int node_count = -1; 
    MPI_( MPI_Recv( &node_count...;
    const int tasks_per_node = numtasks / node_count;


    const int ARRAY_SIZE = 1024 * 1024 * 256; // 256 Mi floats x 2 == 2 GiB total storage
    // @WARNING: ARRAY_SIZE must be evenly divisible by the number of MPI processes
    const int PER_MPI_TASK_ARRAY_SIZE = ARRAY_SIZE / numtasks;
    if( ARRAY_SIZE % numtasks != 0  && task == 0 ) {
        std::cerr << ARRAY_SIZE << " must be evenly divisable by the number of mpi processes" << std::endl;
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    // PER TASK DATA INIT - in the real world this is the place where data are read from file
    // through the MPI_File_ functions or, less likely received from the root process
    std::vector< real_t > v1( ARRAY_SIZE / numtasks, 0. );
    std::vector< real_t > v2( ARRAY_SIZE / numtasks, 0. );
    for( int i = 0; i != PER_MPI_TASK_ARRAY_SIZE; ++i ) {
        v1[ i ] = 1;
        v2[ i ] = 1;  
    }

    real_t partial_dot = 0.;
#ifndef GPU
    for( int i = 0; i != PER_MPI_TASK_ARRAY_SIZE; ++i ) partial_dot += v1[ i ] * v2[ i ];
    std::ostringstream os;
    os << &nodeid[ 0 ] << " - rank: " << task << '\n';
    std::cout << os.str(); os.flush();     
#else
    // SELECT GPU = task % <num gpus on node>, note that with this
    // approach it is possible to support nodes with different numbers of GPUs
    int device_count = 0;
    if( cudaGetDeviceCount( &device_count ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaGetDeviceCount FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    const int device =   ( task / tasks_per_node ) % device_count;
    std::ostringstream os;
    os << &nodeid[ 0 ] << " - rank: " << task << "\tGPU: " << device << '\n';
    std::cout << os.str(); os.flush();     

    if( cudaSetDevice( device ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaGetSetDevice FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    real_t* dev_v1   = 0;
    real_t* dev_v2   = 0;
    real_t* dev_dout = 0;
    if( cudaMalloc( &dev_v1,   sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMalloc FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    if( cudaMalloc( &dev_v2,   sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMalloc FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    if( cudaMalloc( &dev_dout, sizeof( real_t ) * 1 ) != cudaSuccess ) {
        std::cerr << task << ' ' << __LINE__ << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMalloc FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    // MOVE DATA TO GPU
    if( cudaMemcpy( dev_v1, &v1[ 0 ], sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE,
                    cudaMemcpyHostToDevice ) != cudaSuccess ) {
        std::cerr << task << ' ' << __LINE__ << ' ' <<  cudaGetErrorString( cudaGetLastError() ) <<  " cudaMemcpy FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;    
    }
    if( cudaMemcpy( dev_v2, &v2[ 0 ], sizeof( real_t ) * PER_MPI_TASK_ARRAY_SIZE,
                    cudaMemcpyHostToDevice ) != cudaSuccess ) {
        std::cerr << task << ' ' << __LINE__ << ' ' <<  cudaGetErrorString( cudaGetLastError() ) <<  " cudaMemcpy FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }     
    // INVOKE KERNEL
    const int NUM_THREADS_PER_BLOCK = BLOCK_SIZE; // must match size of buffer used for reduction
    const int NUM_BLOCKS = std::min( PER_MPI_TASK_ARRAY_SIZE  / NUM_THREADS_PER_BLOCK,
                                     0xffff ); // max number of blocks is 64k 
    // initialize partial dot product to zero
    if( cudaMemset( dev_dout, 0, sizeof( real_t) ) != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " cudaMemset FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;
    }
    // actual on-device computation    
    dot_product_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>( dev_v1, dev_v2, PER_MPI_TASK_ARRAY_SIZE, dev_dout );
    // check for kernel launch errors: it is not possible to catch on-device execution errors but only
    // if there was an error launching the kernel
    if( cudaGetLastError() != cudaSuccess ) {
        std::cerr << task << ' ' << cudaGetErrorString( cudaGetLastError() ) <<  " kernel launch FAILED\n"; 
        MPI_( MPI_Abort( MPI_COMM_WORLD, 1 ) );
        return 1;      
    }     

    // MOVE DATA TO CPU
    cudaMemcpy( &partial_dot, dev_dout, sizeof( real_t ) * 1, cudaMemcpyDeviceToHost );
#endif

    // REDUCE (SUM) ALL ranks -> rank 0
    real_t result = 0.;
    MPI_( MPI_Reduce( &partial_dot, &result, 1, MPI_REAL_T_, MPI_SUM, 0, MPI_COMM_WORLD ) );

    // IF RANK == 0 -> PRINT RESULT
    if( task == 0 ) {
        std::cout << "dot product result: " << result << std::endl;
    } 
  
#ifdef GPU
    // RELEASE GPU RESOURCES
    cudaFree( dev_v1 );
    cudaFree( dev_v2 );
    cudaFree( dev_dout );
    cudaDeviceReset(); 
#endif

    // RELEASE MPI RESOURCES   
    MPI_( MPI_Finalize() );

    return 0;
}
