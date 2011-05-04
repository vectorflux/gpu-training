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
                                      int num_col ) {
                                      	
  
    return m[ ( blockRow * blockDim.y + row ) * num_col + blockCol * blockDim.x + col ];

}

__global__ void matmul( const real_t* m1, const real_t* m2, real_t* mout,
                        int m1_rows, int m1_columns, int m2_columns  ) { // m1_columns == m2_rows
                                                                         // mout = m1_rows x m2_columns
#ifdef AVOID_BANK_CONFLICTS	
	__shared__ real_t M1[ TILE_SIZE ][ TILE_SIZE ];
	__shared__ real_t M2[ TILE_SIZE ][ TILE_SIZE ];                   
#else
    __shared__ real_t M1[ TILE_SIZE ][ TILE_SIZE     ];
	__shared__ real_t M2[ TILE_SIZE ][ TILE_SIZE + 1 ];     
#endif		
	const int blockRow = blockIdx.y * blockDim.y; 
    const int blockCol = blockIdx.x * blockDim.x;
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    real_t out = 0.f;
    for( int b = 0; b != gridDim.x; ++b ) {
    	//copy data into shared memory
    	M1[ row ][ col ] = get_matrix_element( m1, blockCol + b, blockRow, row, col );
    	M2[ row ][ col ] = get_matrix_element( m2, blockCol, blockRow + b, row, col );
        __syncthreads(); // required to guarantee that data are computed before next step
                         // where a thread accesses data computed by other threads
        for( int k = 0; k != TILE_SIZE; ++k ) {
            out += M1[ row ][ k ] * M2[ k ][ col ];       	
        }
        __synchthreads(); // required to avoid that some threads start modifying
                          // data in cache before all threads have exited for loop    
    }    
}

