// #CSCS CUDA Training 
//
// #Exercise 10 - CUDA 4, peer to peer access, parallel execution on separate GPUs
//
// #Author Ugo Varetto
//
// #Goal: run kernels on separate GPUs passing the same pointer to both kernels; have
//        each kernel operate on a subset of the data
//
// #Rationale: shows how the same memory can be accessed from kernels in separate GPUs and
//             how to time the concurrent execution of kernels
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
// #Compilation: nvcc -arch=sm_20 10_multi-gpu.cu -o multi-gpu
//
// #Execution: ./multi-gpu
//
// #Note: Fermi (2.0) or better required; must be compiled with sm_2x
//
// #Note: Requires at least two GPUs
//
// #Note: timing
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar 
//        rules 

#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>

typedef double real_t;


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


void print_ptr_attr( const cudaPointerAttributes& pa ) {
    std::cout << "\nPointer attributes:\n";
    std::string mt = pa.memoryType == cudaMemoryTypeHost ? "cudaMemoryTypeHost"
                                                         : "cudaMemoryTypeDevice";
    std::cout << "  memoryType:    " << mt << std::endl;
    std::cout << "  device:        " << pa.device << std::endl;
    std::cout << "  devicePointer: " << std::hex << pa.devicePointer << std::endl;
    std::cout << "  hostPointer:   " << pa.hostPointer << std::endl;
}

//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    real_t* dev_buffer = 0;
    const size_t SZ = 550;
    const size_t SIZE = SZ * SZ * SZ;
    const size_t BYTE_SIZE = SIZE * sizeof( real_t );
    int ndev = 0;
    cudaGetDeviceCount( &ndev );
    if( ndev < 2 ) {
        std::cout << "At least two GPU devices required, " << ndev << " found" << std::endl;
    }
   
    std::cout << "\nGrid size: " << BYTE_SIZE / double( 1024 * 1024 * 1024 ) << " GiB" << std::endl;
       
    // on device 0
    cudaSetDevice( 0 );
    cudaMalloc( &dev_buffer, BYTE_SIZE );
    cudaPointerAttributes pointer_attr;
    // print pointer attributes
    cudaPointerGetAttributes( &pointer_attr, dev_buffer );
    print_ptr_attr( pointer_attr );
    // create events for timing 
    cudaEvent_t init_start, init_stop;
    cudaEventCreate( &init_start );
    cudaEventCreate( &init_stop  );
    cudaEventRecord( init_start, 0 );
    // launch initialization kernel on entire grid and time execution 
    init<<< dim3( SZ, SZ, SZ ), 1 >>>( dev_buffer, dim3( SZ, SZ, SZ ), dim3( 0, 0, 0 ) ); 
    cudaEventRecord( init_stop, 0 );
    cudaEventSynchronize( init_stop );
    cudaThreadSynchronize();
    float e;
    cudaEventElapsedTime( &e, init_start, init_stop );
    std::cout << "\nInit elapsed time : " << e << " ms\n" << std::endl;
 
    // switch to device 1
    cudaSetDevice( 1 );
    // print again pointer attribute, just to verify that data are sill valid
    std::cout << "Before cudaDeviceEnablePeerAccess:" << std::endl;
    cudaPointerGetAttributes( &pointer_attr, dev_buffer );
    print_ptr_attr( pointer_attr );
    // enable sharing with device 0
    cudaDeviceEnablePeerAccess( 0, 0 );
    cudaPointerGetAttributes( &pointer_attr, dev_buffer );
    print_ptr_attr( pointer_attr );
    std::cout << "After cudaDeviceEnablePeerAccess:"  << std::endl;
    print_ptr_attr( pointer_attr );
    
    cudaSetDevice( 0 );
    // launch kernel on front part of domain
    cudaEvent_t start1, stop1, start12, stop12, start2, stop2;
    cudaEventCreate( &start1 );
    cudaEventCreate( &start12 );
    cudaEventCreate( &stop1  );
    cudaEventCreate( &stop12  );
    cudaSetDevice( 1 );
    cudaEventCreate( &start2 );
    cudaEventCreate( &stop2  );
    cudaSetDevice( 0 );
    cudaEventRecord( start1, 0 );
    clock_t cpu_start = clock();    
    kernel_on_dev1<<< dim3( SZ, SZ, SZ / 2 ), 1 >>>( dev_buffer, dim3( SZ, SZ, SZ ), dim3( 0, 0, 0 ) );
    cudaSetDevice( 1 );
    // launch kernel on back part of domain
    cudaEventRecord( start2, 0 );
    kernel_on_dev2<<< dim3( SZ, SZ, SZ / 2 ), 1 >>>( dev_buffer, dim3( SZ, SZ, SZ ), dim3( 0, 0, SZ / 2 ) );
    cudaEventRecord( stop2, 0 );
    cudaEventSynchronize( stop2 );
    cudaThreadSynchronize();
    cudaSetDevice( 0 );
    cudaEventRecord( stop1, 0 );
    cudaEventSynchronize( stop1 );
    cudaThreadSynchronize();
    clock_t cpu_end = clock();
    // on POSIX systems CLOCKS_PER_SECOND is always 1E6
    std::cout << "\nCPU time: " << double( cpu_end - cpu_start ) / 1E3 << " ms"<< std::endl;
   
    float e1, e2;
    cudaEventElapsedTime( &e1, start1, stop1 );
    cudaSetDevice( 1 );
    cudaEventElapsedTime( &e2, start2, stop2 );
    cudaSetDevice( 0 );
    std::cout << "GPU time: " <<  std::max( e1, e2 ) << " ms\n" << std::endl;      
    
    std::vector< real_t > host_buffer( SIZE );
    cudaMemcpy( &host_buffer[ 0 ], dev_buffer, BYTE_SIZE, cudaMemcpyDeviceToHost );
    std::cout << ": " << host_buffer.front() << "..." << host_buffer.back() << std::endl; 
    
    
    cudaEventDestroy( init_start );
    cudaEventDestroy( init_stop  );
    cudaEventDestroy( start1 );
    cudaEventDestroy( stop1  );
    cudaFree( dev_buffer );
    cudaSetDevice( 1 );
    cudaEventDestroy( start2 );
    cudaEventDestroy( stop2  );
    
    
    return 0;
}
