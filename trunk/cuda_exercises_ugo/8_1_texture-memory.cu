// WORK IN PROGRESS!!!!

// #CSCS CUDA Training 
//
// #Exercise 8_0 - texture memory
//
// #Author Ugo Varetto
//
// #Goal: compute the sum of two 1D vectors, compare performance of texture vs global
//        memory for storing the arrays
//
// #Rationale: shows how to use texture memory and that texture memory is not faster
//             in cases where input data are not re-used
//
// #Solution: same as ex. 2; add kernel which reads input data from texture memory and
//            properly initialize, map and release texture memory in driver code
//
// #Code: 1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) map texture memory to pre-allocated gpu storage
//        4) copy data from host ro device
//        5) launch kernel
//        6) read data back
//        7) consume data (in this case print result)
//        8) release texture memory 
//        9) free memory
//        
// #Compilation: nvcc -arch=sm_13 8_0_texture-memory.cu -o texture-memory-1
//
// #Execution: ./texture-memory-1 
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

// read input data from global memory
__global__ void apply_stencil( const real_t* gridIn, 
                               const real_t* stencil,
                               real_t* gridOut,
                               int gridNumRows,
                               int gridNumColumns,
                               int stencilSize,
                               int gridSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int stride = gridDim.x * blockDim.x;
    real_t s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i) {
        si = ( gridI + i ) % rows;
        for( j = -halfStencilSize; j <= halfStencilSize; ++j ) {
            sj = ( gridJ + j )
            s += gridIn[ si * stride + sj ] * stencil[ i * stencilSize + j
        }
    }
    gridOut[ gridI * stride + gridJ ] = s;  
}

__global__ void init_grid( real_t* grid ) {
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    grid[ gridI * stride + j ] = real_t( ( gridI + grdJ ) % 2 );                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
}






//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int GRID_NUM_ROWS    = 0x100 + 1; //257
    const int GRID_NUM_COLUMNS = 0x100 + 1; //257
    const int GRID_SIZE = GRID_NUM_ROWS * GRID_NUM_COLUMNS;
    const int GRID_BYTE_SIZE = sizeof( real_t ) * GRID_SIZE;
    const int DEVICE_BLOCK_NUM_ROWS = 4; // num threads per row
    const int DEVICE_BLOCK_NUM_COLUMNS = 4; // num threads per columns
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int DEVICE_GRID_NUM_ROWS    = ( GRID_NUM_ROWS + DEVICE_BLOCK_NUM_ROWS - 1 ) / DEVICE_BLOCK_NUM_ROWS;
    const int DEVICE_GRID_NUM_COLUMNS = ( GRID_NUM_COLUMNS + DEVICE_BLOCK_NUM_COLUMNS - 1 ) / DEVICE_BLOCK_NUM_COLUMNS;
    // if number of threads is not evenly divisable by the number of threads per block 
    // we need an additional block; the above code can be rewritten as
    // if( NUMBER_OF_THREADS % THREADS_PER_BLOCK == 0) BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK;
    // else BLOCK_SIZE = NUMBER_OF_THREADS / THREADS_PER_BLOCK + 1 
 
    //host allocated storage
    std::vector< real_t > host_stencil( STENCIL_SIZE, 1.0f / STENCIL_SIZE );
    std::vector< real_t > host_grid_in( GRID_SIZE );
    std::vector< real_t > host_grid_out( GRID_SIZE );

    // gpu allocated storage
    real_t* dev_grid_in  = 0;
    real_t* dev_grid_out = 0;
    real_t* dev_stencil  = 0;
    cudaMalloc( &dev_grid_in,  GRID_BYTE_SIZE    );
    cudaMalloc( &dev_grid_out, GRID_BYTE_SSIZE   );
    cudaMalloc( &dev_stencil,  STENCIL_BYTE_SIZE );
 
    // copy stencil to device
    cudaMemcpy( &dev_stencil, &host_stencil[ 0 ], STENCIL_BYTE_SIZE, cudaMemcpyDeviceToHost );

    init_grid<<< dim3( GRID_NUM_ROWS, GRID_NUM_COLUMMS, 1), dim3( 1, 1, 1 ) >>>( grid_in );

    // copy initialized grid to host grid, faster than initializing on CPU
    cudaMemcpy( &host_grid_in[ 0 ], &grid_[ in ], GRID_BYTE_SIZE, cudaMamcpyDeviceToHost );

    const dim3 blocks( DEVICE_GRID_NUM_COLUMNS, DEVICE_GRID_NUM_ROWS, 1 );
    const dim3 threads_per_block( DEVICE_BLOCK_NUM_ROWS, DEVICE_BLOCK_NUM_COLUMNS, 1 ); 

    // initialize events for timing execution
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    float e = 0.f;

    cudaEventRecord( start );
    
    // execute kernel accessing global memory
    apply_stencil<<<blocks, threads_per_block>>>( grid_in, grid );
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    // print first and last element of vector
    std::cout << "Result: " << vout[ 0 ] << ".." << vout.back() << std::endl;
    std::cout << "Global memory:  " << e << " ms" << std::endl; 

    cudaEventRecord( start );
    // execute kernel accessing texture memory; input vectors are read from texture references
    sum_vectors_texture<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>( dev_out, VECTOR_SIZE );
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    // print first and last element of vector
    std::cout << "Result: " << vout[ 0 ] << ".." << vout.back() << std::endl;
    std::cout << "Texture memory: " << e << " ms" << std::endl; 

    // release (un-bind/un-map) textures
    cudaUnbindTexture( &v1Tex );
    cudaUnbindTexture( &v2Tex );

    // free memory
    cudaFree( dev_in1 );
    cudaFree( dev_in2 );
    cudaFree( dev_out );

    return 0;
}
