// ******* WORK IN PROGRESS ******* DOES NOT COMPILE


// #CSCS CUDA Training 
//
// #Exercise 5 - block matrix multiply
//
// #Author: Ugo Varetto


#include <cuda.h>
#include <vector>
#include <iostream>

typedef float real_t;

const size_t TILE_SIZE = 16;


__device__ real_t get_matrix_element( const real_t* m, 
                                      int blockCol,
                                      int blockRow,
                                      int col,
                                      int row,
                                      int num_columns ) {                                      	
  
    return m[ ( blockRow * blockDim.y + row ) * num_columns + blockCol * blockDim.x + col ];

}

__global__ void matmul_coalesced( const real_t* m1, const real_t* m2, real_t* mout,
                        int m1_rows, int m1_columns, int m2_columns  ) { // m1_columns == m2_rows
                                                                         // mout = m1_rows x m2_columns
#ifdef AVOID_BANK_CONFLICTS	
	__shared__ real_t M1[ TILE_SIZE ][ TILE_SIZE ];
	__shared__ real_t M2[ TILE_SIZE ][ TILE_SIZE ];                   
#else
    __shared__ real_t M1[ TILE_SIZE ][ TILE_SIZE     ];
	__shared__ real_t M2[ TILE_SIZE ][ TILE_SIZE + 1 ];     
#endif		
	const int blockRow = blockIdx.y; 
    const int blockCol = blockIdx.x;
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    real_t out = 0.f;
    for( int b = 0; b != gridDim.x; ++b ) {
    	//copy data into shared memory
    	M1[ row ][ col ] = get_matrix_element( m1, b, blockRow, row, col, m1_columns );
    	M2[ row ][ col ] = get_matrix_element( m2, blockCol, b, row, col, m2_columns );
        __syncthreads(); // required to guarantee that data are computed before next step
                         // where a thread accesses data computed by other threads
        for( int k = 0; k != TILE_SIZE; ++k ) {
            out += M1[ row ][ k ] * M2[ k ][ col ];       	
        }
        __synchthreads(); // required to avoid that some threads start modifying
                          // data in cache before all threads have exited for loop    
    }
   mout[ ( blockRow * blockDim.y + row ) * m2_columns + blockCol * blockDim.x + col ] = out;     
}

__global__ void matmul( const real_t* m1, const real_t* m2, real_t* mout,
                        int m1_rows, int m1_columns, int m2_columns  ) { // m1_columns == m2_rows
                                                                         // mout = m1_rows x m2_columns
	const int row = blockIdx.y * blockDim.y + threadIdx.x; 
    const int col = blockIdx.x * blockDim.x + threadIdx.y
    const int row = threadIdx.y;
    real_t out = m1[ row * m1_columns + 0 ] * m2[ 0 * m2_columns + col ];

    for( int k = 1; k != m1_columns; ++k ) {
    	out += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
    }
    mout[ row * m2_columns + col ] = out;
}

void matmul_ref( const real_t* m1, const real_t* m2, real_t* mout,
                 int m1_rows, int m1_columns, int m2_columns  ) {
                 	
    for( int row = 0; row != m1_rows; ++row ) {
    	for( int col = 0; col != m2_columns; ++col ) {
    		for( int k = 0; k != m1_columns; ++k ) {
    			mout[ row * m2_columns + col ] = m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
    		}
    	}
    }
}

bool compare( const real_t* v1, const real_t* v2, size_t N, real_t eps { 
	for( int i = 0; i != N; ++i ) {
		if( std::abs( v1[ i ] - v2[ i ] ) > eps ) return false;
	}
}

//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
	
	//1024 x 1024 matrices
	const dim3 BLOCKS( 64, 64 );
	const dim3 THREADS_PER_BLOCK( 16, 16 ); 
	const int ROWS = BLOCKS.y * THREADS_PER_BLOCK.y;
	const int COLUMNS =  BLOCKS.x * THREADS_PER_BLOCK.x;
	const size_t SIZE = ROWS * COLUMNS * sizeof( real_t );
	
	// device storage for gpu computation
	real_t* dev_m1 = 0;
	real_t* dev_m2 = 0;
	real_t* dev_mout = 0;
	cudaMalloc( &dev_m1,  SIZE );
	cudaMalloc( &dev_m2,  SIZE );
	cudaMalloc( &dev_out, SIZE );
	//host storage for reading the output of gpu computation
	std::vector< real_t> host_mout( ROWS * COLUMNS );
	
	// host storage for cpu computation
	std::vector< real_t > m1( ROWS * COLUMNS );
	std::vector< real_t > m2( ROWS * COLUMNS );
	std::vector< real_t > mout( ROWS * COLUMNS );

    // initialize matrix with kernel; much faster than using
    // for loops on the cpu
	init_matrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_m1 );
	init_matrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_m2 );

    
	// print upper 4x4 left corner of input matrix
	std::cout << "INPUT MATRIX 1 - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
	print_matrix( &m[ 0 ], 4, 4, COLUMNS );
	
	// create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop  = cudaEvent_t();
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
    
    // record time into start event 
    cudaEventRecord( start, 0 ); // 0 is the default stream id
	// execute kernel
	transpose<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_in, dev_out, ROWS, COLUMNS );
	//transposeCoalesced<<<BLOCKS, THREADS_PER_BLOCK>>>>( dev_in, dev_out, COLUMNS, ROWS);
    // issue request to record time into stop event
    cudaEventRecord( stop, 0 );
    // synchronize stop event to wait for end of kernel execution on stream 0
    cudaEventSynchronize( stop );
    // compute elapsed time (done by CUDA run-time) 
	float elapsed = 0.f;
	cudaEventElapsedTime( &elapsed, start, stop );
	
	std::cout << "Elapsed time (ms): " << elapsed / 1000 << std::endl;

	// copy output data from device(gpu) to host(cpu)
	cudaMemcpy( &outmatrix[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );

	// print upper 4x4 corner of transposed matrix
	std::cout << "\nOUTPUT MATRIX - " << COLUMNS << " rows, " << ROWS << " columns" << std::endl;
	print_matrix( &outmatrix[ 0 ], 4, 4, ROWS );

    // free memory
    cudaFree( dev_in );
    cudaFree( dev_out );

    // release events
    cudaEventDestroy( start );
	cudaEventDestroy( stop  );


	return 0;
}


