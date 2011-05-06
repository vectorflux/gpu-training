// #CSCS CUDA Training 
//
// #Exercise 3 - transpose matrix
//
// #Author Ugo Varetto
//
// #Goal: compute the transpose of a matrix     
//
// #Rationale: shows how to perform operations on a 2D grid and how to
//             use the GPU for data initializaion
//
// #Solution: straightworwad, simply compute the thread id associated with the element
//            and copy transposed data into output matrix        
//
// #Code: typical flow:
//        1) compute launch grid configuration
//        2) allocate data on host(cpu) and device(gpu)
//        3) initialize data directly on the GPU
//        4) launch kernel
//        5) read data back
//        6) consume data (in this case print result)
//        7) free memory
//        
// #Compilation: nvcc -arch=sm_13 3_0_transpose.cu -o transpose
//
// #Execution: ./transpose
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

typedef float real_t;

__global__ void transpose( const real_t* in, real_t *out, int num_rows, int num_columns ) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_index = row * num_columns + col;
    const int output_index = col * num_rows + row; 
    out[ output_index ] = in[ input_index ];
}


__global__ void init_matrix( real_t* in ) {
    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int r = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = c + gridDim.x * blockDim.x * r; 
    in[ idx ] = (real_t) idx; 
}


void print_matrix( const real_t* m, int r, int c ) {
    for( int i = 0; i != r; ++i ) {
        for( int j = 0; j != c; ++j ) std::cout << m[ i * c + j ] << ' ';
        std::cout << '\n';
    }
    std::cout << std::endl;        
}

//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
    
    const int ROWS = 16;
    const int COLUMNS = 16;
    const dim3 BLOCKS( 4, 4 );
    const dim3 THREADS_PER_BLOCK( 4, 4 ); 
    const size_t SIZE = ROWS * COLUMNS * sizeof( real_t );

    // device(gpu) storage
    real_t* dev_in = 0;
    real_t* dev_out = 0;
    cudaMalloc( &dev_in,  SIZE );
    cudaMalloc( &dev_out, SIZE );

    // host(cpu) storage
    std::vector< real_t > outmatrix( ROWS * COLUMNS );
    
    // initialize data with gpu kernel; faster than CPU for loops
    init_matrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_in );
    cudaMemcpy( &outmatrix[ 0 ], dev_in, SIZE, cudaMemcpyDeviceToHost );

    std::cout << "INPUT MATRIX - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
    print_matrix( &outmatrix[ 0 ], ROWS, COLUMNS );

    // invoke transpose kernel
    transpose<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_in, dev_out, ROWS, COLUMNS );

    // copy output data from device(gpu) to host(cpu)
    cudaMemcpy( &outmatrix[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
    
    // print result
    std::cout << "\nOUTPUT MATRIX - " << COLUMNS << " rows, " << ROWS << " columns" << std::endl;
    print_matrix( &outmatrix[ 0 ], COLUMNS, ROWS );
    
    // free memory
    cudaFree( dev_in );
    cudaFree( dev_out );

    return 0;
}
