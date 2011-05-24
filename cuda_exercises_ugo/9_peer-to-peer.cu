// WORK IN PROGRESS !!!!!!

// #CSCS CUDA Training 
//
// #Exercise 9 - CUDA 4, peer to peer access
//
// #Author Ugo Varetto
//
// #Goal: compare the performance of 2D stencil application with:
//        1) global memory
//        2) texture memory with and without arrays
//        3) shared memory 
//
// #Rationale: shows how texture memory is faster than global memory
//             when data are reused, thanks to (2D) caching; also
//             shows that for periodic boundary conditions using hw wrapping
//             is much faster than performing manual bounds checking
//
// #Solution: implement stencil computation accessing data in global, texture and shared memory;
//            pack double precision data into int2 data types 
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
// #Compilation: nvcc -arch=sm_13 8_2_texture-memory-double-precision.cu -o texture-memory-3
//
// #Execution: ./texture-memory-3 
//
// #warning: texture wrap mode doesn't seem to work with non-power-of-two textures
//
// #Note: textures do not support directly 64 bit (double precision) floating point data 
//        it is however possible unpack doubles into int2 textures and reconstruct the double inside
//        a kernel local variable
//        Global time / Cached time == Cached time / Texture time ~= 2
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
    
    cudaSetDevice( 0 );
    real_t* dev_buffer = 0;
    const size_t SIZE = 1024;
    const size_t BYTE_SIZE = SIZE * sizeof( real_t );
    cudaMalloc( &dev_buffer, BYTE_SIZE );
    kernel_on_dev1<<< SIZE, 1 >>>( dev_buffer );
    std::vector< real_t > host_buffer( SIZE, 0.f );
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Kernel on device 1: " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    cudaSetDevice( 1 );
    kernel_on_dev2<<<  SIZE, 1 >>>( dev_buffer );
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << "Kernel on device 2: " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 

   
    return 0;
}
