// #CSCS CUDA Training 
//
// #Exercise 10 - CUDA 4, peer to peer access, parallel execution on separate GPUs
//
// #Author Ugo Varetto
//
// #Goal: run kernels on separate GPUs passing the same pointer to both kernels
//
// #Rationale: shows how memory can be accessed from kernels in separate GPUs 
//
// #Solution: use setCudaDevice and cudaEnablePeerAccess to select device and
//            enable sharing of memory
//
// #Code: 1) allocate device memory
//        2) select first GPU
//        3) launch kernel
//        4) copy data back from GPU 
//        5) select second GPU
//        6) launch other kernel
//        7) copy data back from GPU 
//        8) free memory
//        
// #Compilation: nvcc -arch=sm_20 9_peer-to-peer.cu -o peer-to-peer
//
// #Execution: ./peer-to-peer
//
// #Note: Fermi (2.0) or better required; must be compiled with sm_2x
//
// #Note: Requires at least two GPUs
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar 
//        rules 
//
// #Note: -arch=sm_13 allows the code to run on every card available on Eiger and possibly even
//        on students' laptops; it's the identifier for the architecture before Fermi (sm_20)
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
//
// #Note: the example can be extended to read configuration data and array size from the command
//        line and could be timed to investigate how performance is dependent on single/double
//        precision and thread block size
#include <cuda.h>
#include <iostream>
#include <vector>

typedef float real_t;


__device__ size_t get_global_index( const dim3& gridSize,
                                    const dim3& offset ) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t yStride = gridSize.x;
    const size_t zStride = yStride * gridSize.y;
    return  ( z + offset.z ) * zStride + ( y + offset.y ) * yStride + x + offset.x;
}


__global__ void kernel_on_dev1( real_t* buffer, dim3 gridSize, dim3 offset ) {
    buffer[ get_global_index( gridSize, offset ) ] =   2.0;  
}

__global__ void kernel_on_dev2( real_t* buffer, dim3 gridSize, dim3 offset ) {
    buffer[ get_global_index( gridSize, offset ) ] =  -2.0;  
}

__global__ void init( real_t* buffer, dim3 gridSize, dim3 offset ) {
    buffer[ get_global_index( gridSize, offset ) ] =   1.0f;
}


#define p std::cout << __LINE__ - 1 << "> " << cudaGetErrorString( cudaGetLastError() ) << std::endl;

//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    real_t* dev_buffer = 0;
    const size_t SZ = 512;
    const size_t SIZE = SZ * SZ * SZ;
    const size_t BYTE_SIZE = SIZE * sizeof( real_t );
    int ndev = 0;
    cudaGetDeviceCount( &ndev );
    if( ndev < 2 ) {
        std::cout << "At least two GPU devices required, " << ndev << " found" << std::endl;
    }
   
    cudaSetDevice( 1 );
    cudaDeviceEnablePeerAccess( 0, 0 );
    // on device 0
    cudaSetDevice( 0 );
    cudaMalloc( &dev_buffer, BYTE_SIZE );
    init<<< dim3( SZ, SZ, SZ ), 1 >>>( dev_buffer, dim3( SZ, SZ, SZ ), dim3( 0, 0, 0 ) );
    cudaThreadSynchronize();

    // launch kernel on front part of domain
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate( &start1 );
    cudaEventCreate( &stop1  );
    cudaEventCreate( &start2 );
    cudaEventCreate( &stop2  );
    cudaEventRecord( start1, 0 );
    kernel_on_dev1<<< dim3( SZ, SZ, SZ / 2 ), 1 >>>( dev_buffer, dim3( SZ, SZ, SZ ), dim3( 0, 0, 0 ) );
    cudaSetDevice( 1 );
    // launch kernel on back part od domain
    cudaEventRecord( start2, 0 );
    kernel_on_dev2<<< dim3( SZ, SZ, SZ / 2 ), 1 >>>( dev_buffer, dim3( SZ, SZ, SZ ), dim3( 0, 0, SZ / 2 ) );
    cudaEventRecord( stop2, 0 );
    cudaEventSynchronize( stop2 );
    cudaThreadSynchronize();
    cudaSetDevice( 0 );
    cudaEventRecord( stop1, 0 );
    cudaEventSynchronize( stop1 );
    cudaThreadSynchronize();

    float e1, e2;
    cudaEventElapsedTime( &e1, start1, stop1 );
    cudaEventElapsedTime( &e2, start2, stop2 );
    std::cout << "Elapsed time (ms): " <<  e1 << ' ' << e2 << std::endl;      
    
    std::vector< real_t > host_buffer( SIZE );
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << ": " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    
    cudaEventDestroy( start1 );
    cudaEventDestroy( start2 );
    cudaEventDestroy( stop1  );
    cudaEventDestroy( stop2  );

    cudaFree( dev_buffer );
    
    return 0;
}
