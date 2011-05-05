// ******* WORK IN PROGRESS *******

// #CSCS CUDA Training 
//
// #Exercise 5 - block matrix multiply
//
// #Author: Ugo Varetto
//
// #Compilation: [standard]       nvcc -arch=sm_13 matmul.cu -o matmul 
//               [block multiply] nvcc -DBLOCK_MULTIPLY -arch=sm_13 matmul.cu -o block_matmul
//
// #Note: try on both G90 and GF100 architectures to verify the impact of L1 cache
 
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

__global__ void block_matmul( const real_t* m1, const real_t* m2, real_t* mout,
                              int m1_columns, int m2_columns  ) { // m1_columns == m2_rows
                                                                      // mout = m1_rows x m2_columns
    __shared__ real_t M1[ TILE_SIZE ][ TILE_SIZE ];
    __shared__ real_t M2[ TILE_SIZE ][ TILE_SIZE ];     
		
	  const int blockRow = blockIdx.y; 
    const int blockCol = blockIdx.x;
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    real_t out = 0.f;
    for( int b = 0; b != m1_columns / TILE_SIZE; ++b ) {
    	  //copy data into shared memory
        M1[ row ][ col ] = get_matrix_element( m1, b, blockRow, col, row, m1_columns );
    	  M2[ row ][ col ] = get_matrix_element( m2, blockCol, b, col, row, m2_columns );
        __syncthreads(); // required to guarantee that data are computed before next step
                         // where a thread accesses data computed by other threads
        for( int k = 0; k != TILE_SIZE; ++k ) {
            out += M1[ row ][ k ] * M2[ k ][ col ];       	
        }
        __syncthreads(); // required to avoid that some threads start modifying
                         // data in cache before all threads have exited for loop    
    }
    mout[ ( blockRow * blockDim.y + row ) * m2_columns + blockCol * blockDim.x + col ] = out;     
}

__global__ void matmul( const real_t* m1, const real_t* m2, real_t* mout,
                        int m1_columns, int m2_columns  ) { // m1_columns == m2_rows
                                                            // mout = m1_rows x m2_columns
    const int row = blockIdx.y * blockDim.y + threadIdx.y; 
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    real_t out = 0.f;//m1[ row * m1_columns + 0 ] * m2[ 0 * m2_columns + col ];

    for( int k = 0; k != m1_columns; ++k ) {
        out += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
    }
    mout[ row * m2_columns + col ] = out;
}

__global__ void init_matrix( real_t* m ) {
    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int r = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = c + gridDim.x * blockDim.x * r; 
    const real_t s = gridDim.x * gridDim.y;
    m[ idx ] = real_t( idx ) / s; 
}


void matmul_ref( const real_t* m1, const real_t* m2, real_t* mout,
                 int m1_rows, int m1_columns, int m2_columns  ) {
                 	
    for( int row = 0; row != m1_rows; ++row ) {
    	  for( int col = 0; col != m2_columns; ++col ) {
          mout[ row * m2_columns + col ] = 0.f; 
    		  for( int k = 0; k != m1_columns; ++k ) {
    			  mout[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
    	    }
    	  }
    }
}

bool compare( const real_t* v1, const real_t* v2, size_t N, real_t eps ) { 
	  for( int i = 0; i != N; ++i ) {
		    if( std::fabs( v1[ i ] - v2[ i ] ) > eps ) return false;
	  }
    return true;
}

void print_matrix( const real_t* m, int r, int c, int stride ) {
  for( int i = 0; i != r; ++i ) {
    for( int j = 0; j != c; ++j )
      std::cout << m[ i * stride + j ] << ' ';
    std::cout << '\n';
  }
  std::cout << std::endl;   
}

//------------------------------------------------------------------------------
int main(int argc, char** argv ) {
	
	//1024 x 1024 matrices
	const dim3 BLOCKS( 64, 64 );
	const dim3 THREADS_PER_BLOCK( 16, 16 ); 
	const int ROWS = BLOCKS.y * THREADS_PER_BLOCK.y;
	const int COLUMNS =  BLOCKS.x * THREADS_PER_BLOCK.x;
	const size_t ARRAY_SIZE = ROWS * COLUMNS;
  const size_t BYTE_SIZE = ARRAY_SIZE * sizeof( real_t );
  	
	// device storage for gpu computation
	real_t* dev_m1 = 0;
	real_t* dev_m2 = 0;
	real_t* dev_mout = 0;
	cudaMalloc( &dev_m1,  BYTE_SIZE );
	cudaMalloc( &dev_m2,  BYTE_SIZE );
	cudaMalloc( &dev_mout, BYTE_SIZE );
	//host storage for reading the output of gpu computation
	std::vector< real_t> host_mout( ARRAY_SIZE );
	
	// host storage for cpu computation
	std::vector< real_t > m1( ARRAY_SIZE );
	std::vector< real_t > m2( ARRAY_SIZE );
	std::vector< real_t > mout( ARRAY_SIZE );

  // initialize matrix with kernel; much faster than using
  // for loops on the cpu
	init_matrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_m1 );
	init_matrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_m2 );
 
  // copy initialized data into host arrays for further processing on the gpu
  cudaMemcpy( &m1[ 0 ], dev_m1, BYTE_SIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( &m2[ 0 ], dev_m2, BYTE_SIZE, cudaMemcpyDeviceToHost );
   
	// print upper 4x4 left corner of input matrix 1
	std::cout << "INPUT MATRIX 1 - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
	print_matrix( &m1[ 0 ], 4, 4, COLUMNS );
  // print upper 4x4 left corner of input matrix 2
  std::cout << "INPUT MATRIX 2 - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
  print_matrix( &m2[ 0 ], 4, 4, COLUMNS );
	
	// create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop  = cudaEvent_t();
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
    
  // record time into start event 
  cudaEventRecord( start, 0 ); // 0 is the default stream id

#ifdef BLOCK_MULTIPLY	
  // execute kernel
	block_matmul<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_m1, dev_m2, dev_mout,  COLUMNS, COLUMNS );
#else  
  matmul<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_m1, dev_m2, dev_mout,  COLUMNS, COLUMNS );
#endif  
  
  // issue request to record time into stop event
  cudaEventRecord( stop, 0 );
  // synchronize stop event to wait for end of kernel execution on stream 0
  cudaEventSynchronize( stop );
  // compute elapsed time (done by CUDA run-time) 
	float elapsed = 0.f;
	cudaEventElapsedTime( &elapsed, start, stop );
	
	std::cout << "Elapsed time (ms): " << elapsed / 1000 << std::endl;

	// copy output data from device(gpu) to host(cpu)
	cudaMemcpy( &host_mout[ 0 ], dev_mout, BYTE_SIZE, cudaMemcpyDeviceToHost );

	// print upper 4x4 corner of output matrix
	std::cout << "\nGPU OUTPUT MATRIX - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
	print_matrix( &host_mout[ 0 ], 4, 4, COLUMNS );

  // compute on cpu
  matmul_ref( &m1[ 0 ], &m2[ 0 ], &mout[ 0 ], ROWS, COLUMNS, COLUMNS );  
  // print upper 4x4 corner of output matrix
  std::cout << "\nCPU OUTPUT MATRIX - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
  print_matrix( &mout[ 0 ], 4, 4, COLUMNS );

#ifdef COMPARE_RESULTS
  // warning: requires real_t = double to pass
  std::cout << "Comparing... ";
  if( compare( &host_mout[ 0 ], &mout[ 0 ], ARRAY_SIZE, 0.00001 ) ) std::cout << "PASSED" << std::endl;
  else std::cout << "FAILED" << std::endl;
#endif

  // free memory
  cudaFree( dev_m1 );
  cudaFree( dev_m2 );
  cudaFree( dev_mout );

  // release events
  cudaEventDestroy( start );
  cudaEventDestroy( stop  );

	return 0;
}


