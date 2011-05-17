// #CSCS CUDA Training 
//
// #Exercise 8_1 - texture memory, 2D stencil
//
// #Author Ugo Varetto
//
// #Goal: compare the performance of 2D stencil application with:
//        1) global memory
//        2) texture memory
//        3) shared memory 
//
// #Rationale: shows how texture memory is faster than global memory
//             when data are reused, thanks to (2D) caching; also
//             shows that for periodic boundary conditions using hw wrapping
//             is much faster than performing manual bounds checking
//
// #Solution: implement stencil computation accessing data in global, texture and shared memory
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
// #Compilation: nvcc -arch=sm_13 8_1_texture-memory.cu -o texture-memory-2
//
// #Execution: ./texture-memory-2 
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
                               int stencilSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int soff = halfStencilSize;
    real_t s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i) {
        si = gridI + i;
        if( si < 0 ) si += gridNumRows;
        else if( si >= gridNumRows ) si -= gridNumRows;
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
            sj = gridJ + j;
            if( sj < 0 ) sj += gridNumColumns;
            else if( sj >= gridNumColumns ) sj -= gridNumColumns;
            s += gridIn[ si * gridNumColumns + sj ] * stencil[ ( i + soff ) * stencilSize + ( j + soff ) ];
        }
    }
    gridOut[ gridI * gridNumColumns + gridJ ] = s;
}

texture< real_t, 2 > gridInTex;
texture< real_t, 2 > stencilTex;


// read input data from global memory
__global__ void apply_stencil_texture( real_t* gridOut,
                                       int gridNumRows,
                                       int gridNumColumns,
                                       int stencilSize ) {
    // compute current thread id
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfStencilSize = stencilSize / 2;
    if( gridI >= gridNumRows || gridJ >= gridNumColumns ) return;
    const int soff = halfStencilSize;
    real_t s = 0.f; 
    int si = 0;
    int sj = 0;
    for( int i = -halfStencilSize; i <= halfStencilSize; ++i) {
        si = gridI + i;
#ifndef TEXTURE_WRAP
        if( si < 0 ) si += gridNumRows;
        else if( si >= gridNumRows ) si -= gridNumRows;
#endif              
        for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
             sj = gridJ + j;
#ifndef TEXTURE_WRAP
             if( sj < 0 ) sj += gridNumColumns;
             else if( sj >= gridNumColumns ) sj -= gridNumColumns;
#endif                               
             s += tex2D( gridInTex, sj, si ) * 
                  tex2D( stencilTex, j + soff, i + soff );
        }
    }
    gridOut[ gridI * gridNumColumns + gridJ ] = s;
}


void apply_stencil_ref( const real_t* gridIn,
                        const real_t* stencil,
                        real_t* gridOut,
                        int gridNumRows,
                        int gridNumColumns,
                        int stencilSize ) {
                                                    
     const int halfStencilSize = stencilSize / 2;
     const int soff = halfStencilSize;
     for( int r = 0; r != gridNumRows; ++r ) {
         for( int c = 0; c != gridNumColumns; ++c ) {
             real_t s = 0.f; 
             int si = 0;
             int sj = 0;
             for( int i = -halfStencilSize; i <= halfStencilSize; ++i) {
                 si = r + i;
                 if( si < 0 ) si += gridNumRows;
                 else if( si >= gridNumRows ) si -= gridNumRows;
                 for( int j = -halfStencilSize; j <= halfStencilSize; ++j ) {
                      sj = c + j;
                      if( sj < 0 ) sj += gridNumColumns;
                      else if( sj >= gridNumColumns ) sj -= gridNumColumns;
                     s += gridIn[ si * gridNumColumns + sj ] * stencil[ ( i + soff ) * stencilSize + ( j + soff ) ];
                 }
             }     
             gridOut[ r * gridNumColumns + c ] = s;
         }
     }
}

__global__ void init_grid( real_t* grid ) {
    const int gridI = blockIdx.y * blockDim.y + threadIdx.y;
    const int gridJ = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    grid[ gridI * stride + gridJ ] = real_t( ( gridI + gridJ ) % 2 );                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
}


//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int GRID_NUM_ROWS    = 0x100;// + 1; //257
    const int GRID_NUM_COLUMNS = 0x100;// + 1; //257
    const int GRID_SIZE = GRID_NUM_ROWS * GRID_NUM_COLUMNS;
    const int GRID_BYTE_SIZE = sizeof( real_t ) * GRID_SIZE;
    const int DEVICE_BLOCK_NUM_ROWS = 4; // num threads per row
    const int DEVICE_BLOCK_NUM_COLUMNS = 4; // num threads per columns
    const int STENCIL_EDGE_LENGTH = 3;
    const int STENCIL_SIZE = STENCIL_EDGE_LENGTH * STENCIL_EDGE_LENGTH;
    const int STENCIL_BYTE_SIZE = sizeof( real_t ) * STENCIL_SIZE;
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to NUMBER_OF_THREADS 
    const int DEVICE_GRID_NUM_ROWS    = ( GRID_NUM_ROWS    + DEVICE_BLOCK_NUM_ROWS    - 1 ) / DEVICE_BLOCK_NUM_ROWS;
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
    cudaMalloc( &dev_grid_out, GRID_BYTE_SIZE   );
    cudaMalloc( &dev_stencil,  STENCIL_BYTE_SIZE );
 
    // copy stencil to device
    cudaMemcpy( dev_stencil, &host_stencil[ 0 ], STENCIL_BYTE_SIZE, cudaMemcpyHostToDevice );

    init_grid<<< dim3( GRID_NUM_ROWS, GRID_NUM_COLUMNS, 1), dim3( 1, 1, 1 ) >>>( dev_grid_in );

    // copy initialized grid to host grid, faster than initializing on CPU
    cudaMemcpy( &host_grid_in[ 0 ], dev_grid_in, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );

    const dim3 blocks( DEVICE_GRID_NUM_COLUMNS, DEVICE_GRID_NUM_ROWS, 1 );
    const dim3 threads_per_block( DEVICE_BLOCK_NUM_COLUMNS, DEVICE_BLOCK_NUM_ROWS, 1 ); 

    //--------------------------------------------------------------------------
    // initialize events for timing execution
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    float e = 0.f;

    cudaEventRecord( start );
    
    // execute kernel accessing global memory
    apply_stencil<<<blocks, threads_per_block>>>( dev_grid_in,
                                                  dev_stencil,
                                                  dev_grid_out,
                                                  GRID_NUM_ROWS,
                                                  GRID_NUM_COLUMNS,
                                                  STENCIL_EDGE_LENGTH );
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // read back result
    cudaMemcpy( &host_grid_out[ 0 ], dev_grid_out, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    // print grid
    std::cout << "Result: " << host_grid_out.front() << ".." << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms" << std::endl; 

    //--------------------------------------------------------------------------
    // describe data inside texture: 1-component floating point value in this case    
    const int BITS_PER_BYTE = 8;
    cudaChannelFormatDesc cd = cudaCreateChannelDesc( sizeof( real_t ) *  BITS_PER_BYTE,
                                                      0, 0, 0, cudaChannelFormatKindFloat );
#ifdef TEXTURE_WRAP    
    gridInTex.addressMode[ 0 ] = cudaAddressModeWrap;
    gridInTex.addressMode[ 1 ] = cudaAddressModeWrap;
#endif                                                      
    // bind textures to pre-allocated storage
    int texturePitch = sizeof( real_t ) * GRID_NUM_COLUMNS;
    cudaBindTexture2D( 0, &gridInTex,   dev_grid_in, &cd, GRID_NUM_COLUMNS, GRID_NUM_ROWS, texturePitch );
    texturePitch = sizeof( real_t ) * STENCIL_EDGE_LENGTH;
    cudaBindTexture2D( 0, &stencilTex,  dev_stencil, &cd, STENCIL_EDGE_LENGTH, STENCIL_EDGE_LENGTH, texturePitch );                                                  

    cudaEventRecord( start );

    // execute kernel accessing global memory
    apply_stencil_texture<<<blocks, threads_per_block>>>( dev_grid_out,
                                                          GRID_NUM_ROWS,
                                                          GRID_NUM_COLUMNS,
                                                          STENCIL_EDGE_LENGTH );
    
    cudaEventRecord( stop );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &e, start, stop );
    // release texture
    cudaUnbindTexture( &gridInTex  );
    cudaUnbindTexture( &stencilTex );
    // read back result
    cudaMemcpy( &host_grid_out[ 0 ], dev_grid_out, GRID_BYTE_SIZE, cudaMemcpyDeviceToHost );
    // print grid
    std::cout << "Result: " << host_grid_out.front() << ".." << host_grid_out.back() << std::endl;
    std::cout << "Time:   " << e << " ms" << std::endl; 

    //--------------------------------------------------------------------------
    apply_stencil_ref( &host_grid_in[ 0 ],
                       &host_stencil[ 0 ],
                       &host_grid_out[ 0 ],
                       GRID_NUM_ROWS,
                       GRID_NUM_COLUMNS,
                       STENCIL_EDGE_LENGTH );
    std::cout << "Result: " << host_grid_out.front() << ".." << host_grid_out.back() << std::endl;

    

    // free memory
    cudaFree( dev_grid_in );
    cudaFree( dev_grid_out );
    cudaFree( dev_stencil );

    return 0;
}
