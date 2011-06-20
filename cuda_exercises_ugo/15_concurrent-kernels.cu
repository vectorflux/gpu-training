// !!!! WORK IN PROGRESS !!!

// #CSCS CUDA Training 
//
// #Example 15 - concurrent kernels
//
// #Author Ugo Varetto
//
// #Goal: compute the sum of two vectors overlapping communication and computation
//
// #Rationale: using streams it is possible to subdivide computation and memory transfer
//             operations in separate execution queues which can be executed in parallel;
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
//           being processed by separate streams is VECTOR_CHUNK_SIZE x NUMBER_OF_STREAMS
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
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like cudaMemcpy(...,cudaDeviceToHost)
//        kernel execution is guaranteed to be terminated before data are copied   
//
// #Note: the code is C++ also because the default compilation mode for CUDA is C++, all functions
//        are named with C++ convention and the syntax is checked by default against C++ grammar rules 
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 (gtx 2xx) or better
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision
//
// #Note: the example can be extended to read configuration data and array size from the command line and
//        investigating the optimal configuration for number of streams and chunk size


//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>

__global__ void timed_kernel( clock_t* clocksElement, int execTime ) {
    const clock_t start = clock();
    clock_t elapsed;
    do {
        elapsed = clock() - start;
    } while( elapsed < execTime );
    *clocksElement = elapsed;
}


__global__ sum_clocks( clock_t* result, const clock_t* clocks, int numElements ) {
    
    __shared__ cache[ CACHE_SIZE ];
    cache[ threadIdx.x ] = 0;
    for( int i = 0; i < numElemens; i += CACHE_SIZE ) {
        cache[ threadIdx.x ] += clocks[ threadIdx.x + i ];   
    }
    __syncthreads();
    for( int i = CACHE_SIZE / 2; i > 0; i /= 2 ) {
        if( threadIdx.x < i ) cache[ threadIdx.x ] += cache[ threadIdx + i ];
        __syncthreads();
    }        
    *result = cache[ 0 ];
}




//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    //first task: verify support for concurrent kernel execution
    cudaDeviceProp prop = cudaDeviceProp();
    int currentDevice = -1;
    cudaGetDevice( &currentDevice );
    cudaGetDeviceProperties( &prop, currentDevice );
    if( prop.concurrentKernels == 0 ) {
        std::cout << "Concurrent kernel execution not supported\n"
                  << "kernels will be serialized" << std::endl;
    }    

    const int NUM_KERNELS = 4;
    const int NUM_STREAMS = NUM_KERNELS;
    const int NUM_CLOCKS  = NUM_KERNELS;
    const size_t CLOCKS_BYTE_SIZE = NUM_CLOCKS * sizeof( clock_t );
    const int KERNEL_EXECUTION_TIME = 250; //ms
    float elapsed_time = 0.f;   
    cudaEvent_t start, stop;
    std::vector< cudaEvent_t >  kernel_events( NUM_KERNELS );
    cidaStream_t sync_stream;
    std::vector< cudaStream_t > kernel_streams( NUM_KERNELS );

    //create timing events
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );

    //create kernel events
    for( std::vector< cudaEvent_t >::iterator i =  kernel_events.begin();
         i != kernels_events.end(); ++i ) {
        cudaEventCreateWithFlags( &(*i), cudaEventDisableTiming );             
        
    }

    //create sync stream: sync stream wait for all kernel events to be recorded 
    cudaStreamCreate( &sync_stream );
    
    //create kernel streams
    for( std::vector< cudaStream_t >::iterator i =  kernel_streams.begin();
         i != kernels_streams.end(); ++i ) {
        cudaStreamCreate( &(*i) );           
    }

    //data array to hold timing information from kernel runs; TODO: use std::vector with page locked allocator
    clock_t* clock_sum = 0;
    //we need host-allocated page locked memory because later-on an async memcpy operation is
    //is used; async operations *always* require page-locked memory
    cudaMallocHost( &clock_sum, sizeof( clock_t ) );
    clock_t* dev_clocks = 0;
    cudaMalloc( &dev_clocks, CLOCKS_BYTE_SIZE );
    clock_t* dev_clock_sum = 0;
    cudaMalloc( &dev_clock_sum, sizeof( clock_t ) );


    // BEGIN of async operations
    cudaEventRecord( start, 0 );
    for( int k = 0; k != NUM_KERNELS; ++k ) {
        const int CLOCK_FREQ_kHz = prop.clockRate; // 1000 * fHz --> T = tms * prop.clockRate
        timed_kernel<<< 1, 1, 0, kernel_streams[ i ] >>>( dev_clocks, kernel_time * CLOCK_FREQ_kHz );
        cudaEventRecord( kernel_events[ i ], kernel_streams[ i ] );
        cudaStreamWaitEvent( sync_stream, kernel_events[ i ] );
    }

    sum_clocks<<< 1, 16, 0, sync_stream >>>( dev_clock_sum, dev_clocks, num_kernels );
    cudaMemcpyAsync( clock_sum, dev_clock_sum, sizeof( clock_t ), cudaMemcpyDeviceToHost );
    //record event, not associated with any stream and therefore recorded
    //after *all* stream events are recorded
    cudaEventRecord( stop, 0 );

    // END of async operations
    
    //sync everything
    //this synchronization call forces to wait until the stop event has been recorded;
    //the stop event is associated with the global context (the '0' in the cudaEventRegister call)
    //and therefore all events in the context must have been recorded before the stop event is recorded
    cudaEventSynchronize( stop );
    cudaElapsedTime( start, stop, &elapsed_time );    
   
//TODO: PRINT SOMETHING OUT!!!

    //free resources
    for( std::vector< cudaEvent_t >::iterator i =  kernel_events.begin();
         i != kernels_events.end(); ++i ) {
        cudaEventDestroy( *i );            
    }

    //create sync stream: sync stream wait for all kernel events to be recorded 
    cudaStreamDestroy( sync_stream );
    
    //create kernel streams
    for( std::vector< cudaStream_t >::iterator i =  kernel_streams.begin();
         i != kernels_streams.end(); ++i ) {
        cudaStreamDestroy( *i );           
    }

    cudaFreeHost( clock_sum );
    cudaFree( dev_clocks );
    cudaFree( dev_clock_sum );

    //OPTIONAL, must be called in order for profiling and tracing tools
    //to show complete traces
    cudaDeviceReset(); 

    return 0;
}

}
