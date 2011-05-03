// ******* WORK IN PROGRESS *******

// #CSCS CUDA Training 
//
// #Exercise 4 - dot product
//
// #Author: Ugo Varetto
//
// #Goal: compute the dot product of two vectors 
//
// #Rationale: shows how to perform the dot product of two vectors as a parallel reduction
// 
// #Solution: store scalar products in local cache and iterate over cache elements
//            performing incremental sums 
//
// #Code: 1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) copy data from host ro device
//        4) create events
//        5) record start time
//        6) launch kernel
//        7) synchronize events to guarantee that kernel execution is finished
//        8) record stop time
//        9) read data back 
//        10) print timing information as stop - start time 
//        11) delete events 
//        12) free memory      
//        The code uses the default stream 0; streams are used to sychronize operations
//        to guarantee that all operations in the same stream are executed sequentially.
//             
// #Compilation: nvcc -arch=sm_13 device-query.cu -o dot_product
//
// #Execution: ./dot_product
//
// #Note: shows how parallel floating point operations lead to unpredictable results
//
// #Note: shared memory *must* be allocated at compile time, on the same OpenCL allows
//        client code to specify size of shared memory at kernel invocation time 
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card available on Eiger and possibly even
//        on students' laptops; it's the identifier for the architecture before Fermi (sm_20)
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
//
// #Note: the example can be extended to read configuration data and matrix size from the command line

#include <cuda.h>
#include <vector>
#include <iostream>
#include <numeric>

typedef float real_t;

const size_t BLOCK_SIZE = 256;

__global__ void partial_dot( const real_t* v1, const real_t* v2, real_t* out, int N ) {
	__shared__ real_t cache[ BLOCK_SIZE ];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[ threadIdx.x ] = 0.f;
    while( i < N ) {
        cache[ threadIdx.x ] += v1[ i ] * v2[ i ];
        i += gridDim.x * blockDim.x;
    }    
    i = BLOCK_SIZE / 2;
    while( i > 0 ) {
    	if( threadIdx.x < i ) cache[ threadIdx.x ] = cache[ threadIdx.x + i ];
    	__syncthreads();
    	i /= 2; //not sure bitwise operations are indeed faster
    }

    if( threadIdx.x == 0 ) out[ blockIdx.x ] = cache[ 0 ];
}

__global__ void init_vector( real_t* v, int N ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while( i < N ) {
		v[ i ] = 1.f; //real_t( i ) / 1000000.f;
		i += gridDim.x * blockDim.x;
	} 
}


//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
	
	const size_t ARRAY_SIZE = 1024 * 1024; //1Mi elements
	const int BLOCKS = 512;
	const int THREADS_PER_BLOCK = 256; // total threads = 512 x 256 = 128ki threads;
	                                   // each thread spans 8 array element  
	const size_t SIZE = ARRAY_SIZE * sizeof( real_t );
	
	// device storage
	real_t* dev_v1 = 0; // vector 1
	real_t* dev_v2 = 0; // vector 2
	real_t* dev_vout = 0; // partial redution = number of blocks
	cudaMalloc( &dev_v1,  SIZE );
	cudaMalloc( &dev_v2,  SIZE );
	cudaMalloc( &dev_vout, BLOCKS * sizeof( real_t ) );
	
	// initialize vector 1 with kernel; much faster than using for loops on the cpu
	init_vector<<< 1024, 256  >>>( dev_v1, ARRAY_SIZE );
    // initialize vector 2 with kernel; much faster than using for loops on the cpu
	init_vector<<< 1024, 256  >>>( dev_v2, ARRAY_SIZE );
	
	// create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop  = cudaEvent_t();
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
    
    // record time into start event 
    cudaEventRecord( start, 0 ); // 0 is the default stream id
	// execute kernel
	partial_dot<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_v1, dev_v2, dev_vout, ARRAY_SIZE );
	// issue request to record time into stop event
    cudaEventRecord( stop, 0 );
    // synchronize stop event to wait for end of kernel execution on stream 0
    cudaEventSynchronize( stop );
    // compute elapsed time (done by CUDA run-time) 
	float elapsed = 0.f;
	cudaEventElapsedTime( &elapsed, start, stop );
	
	std::cout << "Elapsed time (ms): " << elapsed / 1000 << std::endl;

	// copy output data from device(gpu) to host(cpu)
	std::vector< real_t > vout( BLOCKS );
	cudaMemcpy( &vout[ 0 ], dev_vout, BLOCKS * sizeof( real_t ), cudaMemcpyDeviceToHost );

	// print dot product by summing the partially reduced vectors
	std::cout << std::accumulate( vout.begin(), vout.end(), real_t( 0 ) ) << std::endl;	

    // free memory
    cudaFree( dev_v1 );
    cudaFree( dev_v2 );
    cudaFree( dev_vout );

    // release events
    cudaEventDestroy( start );
	cudaEventDestroy( stop  );


	return 0;
}
