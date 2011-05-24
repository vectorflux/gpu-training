// #CSCS CUDA Training 
//
// #Exercise 9 - CUDA 4, peer to peer access
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

__global__ void kernel_on_dev1( real_t* buffer ) {
    buffer[ blockIdx.x ] = 3.0;  
}

__global__ void kernel_on_dev2( real_t* buffer ) {
    buffer[ blockIdx.x ] *= 2.0;  
}

//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    real_t* dev_buffer = 0;
    const size_t SIZE = 1024;
    const size_t BYTE_SIZE = SIZE * sizeof( real_t );
    // on device 0
    cudaSetDevice( 0 );
    cudaMalloc( &dev_buffer, BYTE_SIZE );
    kernel_on_dev1<<< SIZE, 1 >>>( dev_buffer );
    std::vector< real_t > host_buffer( SIZE );
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Kernel on device 1: " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    // on device 1
    cudaSetDevice( 1 );
    const int PEER_DEVICE_TO_ACCESS = 0;
    const int PEER_ACCESS_FLAGS = 0; // reserved for future use, must be zero
    cudaDeviceEnablePeerAccess( PEER_DEVICE_TO_ACCESS, PEER_ACCESS_FLAGS ); // <- enable current device(1) to access device 0
    kernel_on_dev2<<<  SIZE, 1 >>>( dev_buffer );
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Kernel on device 2: " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    cudaFree( dev_buffer );
    return 0;
}
