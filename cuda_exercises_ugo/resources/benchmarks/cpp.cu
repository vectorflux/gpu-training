// #Benchmark 1 - C++ stencil operations
//
// #Author Ugo Varetto
//
// #Goal: implement 2D stencil operations in such a way that the stencil operation to perform
//        is passed as a paramter to the GPU kernel
//
// #Rationale: CUDA allows to use a subset of C++ in kernels; this is useful to write
//             reusable code with operations which can be split into separate classes and
//             composed as needed from withing the kernel
//
// #Solution: split stencil operations into:
//            . an application function: data accessor, operation -> application function
//            . a data access function: 2D coordinate -> (reference to) element
//            . and a stencil operator: accessor -> stencil operation result
//
// #Code: 
//        1) compute launch grid configuration
//        2) init data on device
//        3) launch kernel
//        5) read data back
//        6) consume data (in this case print result)
//        7) free memory
//        
// #Compilation: [default] nvcc -arch=sm_13 cpp.cu -o cpp
//               [loops in c++ kernel ] -DGENERIC_OP
//               [no bounds checking  ] -NO_BOUND_CHECK 
//
// #Execution: ./cpp
//
// @Note: IN PROGRESS
//
// #Note: experiment with different #define switches combinations
//
// #Note: kernel invocations ( foo<<<...>>>(...) ) are *always* asynchronous and a call to 
//        cudaThreadSynchronize() is required to wait for the end of kernel execution from
//        a host thread; in case of synchronous copy operations like 
//        cudaMemcpy(...,cudaDeviceToHost) kernel execution is guaranteed to be terminated before
//        data are copied       
//
// #Note: -arch=sm_13 allows the code to run on every card with hw architecture GT200 
//        (gtx 2xx) or better
//
// #Note: -arch=sm_13 is the lowest architecture version that supports double precision


//#include <cuda_runtime.h> // automatically added by nvcc
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <ctime>

typedef float real_t;

//========================== Utility ===========================================

void print2d( const real_t* in, int columns, int maxrow = 8, int maxcol = 8 ) {
    std::cout << std::endl;
    for( int r = 0; r != maxrow; ++r ) {
        for( int c = 0; c != maxcol; ++c ) {
            std::cout << in[ columns * r + c ] << ' ';
        }
        std::cout << std::endl;
    }
}


//------------------------------------------------------------------------------
// return global 1d index from 2d index + offset; properly handle out of bound
// indices (in this case by clamping to value on edge)
__device__ size_t get_global_idx_2d( int rowOffset = 0, int colOffset = 0 ) {
    const int gridWidth  = gridDim.x * blockDim.x;
    const int gridHeight = gridDim.y * blockDim.y;
    int row    = blockIdx.y * blockDim.y + threadIdx.y + rowOffset;
    int column = blockIdx.x * blockDim.x + threadIdx.x + colOffset;
#ifndef NO_BOUND_CHECK
    // clamp to edge 
    if( row >= gridHeight ) row = gridHeight - 1;
    else if( row < 0 ) row = 0;
    if( column >= gridWidth ) column = gridWidth - 1;
    else if( column < 0 ) column = 0;
#endif          
    return  row * gridWidth + column;
}

//------------------------------------------------------------------------------
// return global 1d index from 2d index + offset; in this case the 2d index
// is assumed to be always within the grid core space (whole grid - halo regions)
__device__ size_t get_global_idx_2d_core_space( int gridWidth, int rowOffset = 0, 
                                                int colOffset = 0 ) {
    const int row    = blockIdx.y * blockDim.y + threadIdx.y + rowOffset;
    const int column = blockIdx.x * blockDim.x + threadIdx.x + colOffset;
    return  row * gridWidth + column;
}

__global__ void memcpy_2d_kernel( const real_t* in, real_t* out,
                                  int rowOffset, int colOffset,
                                  int inNumColumns,
                                  int numRows, int numColumns ) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( row < numRows && col < numColumns ) {
        const int inIdx = ( row + rowOffset ) * inNumColumns + ( col + colOffset );
        const int outIdx = row * numColumns + col;    
        out[ outIdx ] = in[ inIdx ];
    }                    
}

// copy subregion of device memory to host
cudaError_t memcpy_2d_device_to_host( real_t* host_out, real_t* dev_in, 
                                      int inRows, int inColumns,
                                      int inRowOffset, int inColOffset, 
                                      int numRows, int numColumns ) {                       
    cudaError_t err;
    real_t* dev_out = 0;
    err = cudaMalloc( &dev_out, sizeof( real_t ) * numRows * numColumns );
    if( err != cudaSuccess ) return err;
    const dim3 threads( 16, 16, 1 );
    const dim3 blocks( ( numColumns + threads.x - 1 ) / threads.x,
                       ( numRows    + threads.y - 1 ) / threads.y, 1 );
    memcpy_2d_kernel<<< blocks, threads >>>( dev_in, dev_out, inRowOffset, inColOffset, inColumns,
                                             numRows, numColumns );
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        cudaFree( dev_out );
        return err;
    }
    err = cudaMemcpy( host_out, dev_out, sizeof( real_t ) * numRows * numColumns,
                      cudaMemcpyDeviceToHost );
    if( err != cudaSuccess ) {
        cudaFree( dev_out );
        return err;
    }
    return cudaFree( dev_out );
}


//============================  Types ==========================================

//------------------------------------------------------------------------------
template < typename T > class In2DAccessor {
public:
    typedef T element_type;
    typedef const element_type* pointer_type;
    typedef const element_type& reference_type;
    __host__ In2DAccessor( pointer_type grid ) : grid2D_( grid ) {}
    __device__ reference_type operator()( int i = 0, int j = 0 ) const {
        return grid2D_[ get_global_idx_2d( i, j ) ];
    }
private:
    pointer_type grid2D_;          
};

//------------------------------------------------------------------------------
template < typename T > class Out2DAccessor {
public:
    typedef T element_type;
    typedef element_type* pointer_type;
    typedef element_type& reference_type;
    __host__ Out2DAccessor( pointer_type grid ) : grid2D_( grid ) {}
    __device__ reference_type operator()( int i = 0, int j = 0 ) {
        return grid2D_[ get_global_idx_2d( i, j ) ];
    }
private:
    pointer_type grid2D_;             
};

//------------------------------------------------------------------------------
template < typename T, typename StencilOperatorT > class In2DCoreAccessor {
public:
    typedef T element_type;
    typedef const element_type* pointer_type;
    typedef const element_type& reference_type;
    __host__ In2DCoreAccessor( pointer_type grid, int fullGridWidth ) 
        : grid2D_( grid ), fullGridWidth_( fullGridWidth )  {}
    __device__ reference_type operator()( int i = 0, int j = 0 ) const {
        return grid2D_[ 
            get_global_idx_2d_core_space( fullGridWidth_, 
                i + StencilOperatorT::CORE_SPACE_ROW_OFFSET,
                j + StencilOperatorT::CORE_SPACE_COL_OFFSET ) ];
    }
private:
    pointer_type grid2D_;
    int fullGridWidth_; // grid width = core space width + halo width
};

//------------------------------------------------------------------------------
template < typename T, typename StencilOperatorT > class Out2DCoreAccessor {
public:
    typedef T element_type;
    typedef element_type* pointer_type;
    typedef element_type& reference_type;
    __host__ Out2DCoreAccessor( pointer_type grid, int fullGridWidth ) 
        : grid2D_( grid ), fullGridWidth_( fullGridWidth )  {} 
    __device__ reference_type operator()( int i = 0, int j = 0 ) {
        return grid2D_[ 
            get_global_idx_2d_core_space( fullGridWidth_, 
                i + StencilOperatorT::CORE_SPACE_ROW_OFFSET,
                j + StencilOperatorT::CORE_SPACE_COL_OFFSET ) ];
    }
private:
    pointer_type grid2D_;
    int fullGridWidth_; // grid width = core space width + halo width             
};


//------------------------------------------------------------------------------
template < int width, int height > struct StencilOperator {
    enum { CORE_SPACE_ROW_OFFSET = height / 2, CORE_SPACE_COL_OFFSET = width / 2 };
    enum { WIDTH = width, HEIGHT = height };
    enum { WIDTH_MIN_OFFSET = -width / 2, WIDTH_MAX_OFFSET = width / 2,
           HEIGHT_MIN_OFFSET = -height / 2, HEIGHT_MAX_OFFSET = height / 2 };
    enum { AREA = width * height };       
}; 

//------------------------------------------------------------------------------
template < typename T, int width, int height > struct Average : StencilOperator< width, height > {
    template < typename In2DAccessor >
    __device__ T operator()( const In2DAccessor& a ) {
         const real_t W = 1.f / AREA;
         T out = T();
         for( int i = HEIGHT_MIN_OFFSET; i <= HEIGHT_MAX_OFFSET; ++i ) {
             for( int j = WIDTH_MIN_OFFSET; j <= WIDTH_MAX_OFFSET; ++j ) {
                 out += a( i, j ) * W; // ideally the loop nest should be in the base class
             }        
         }
         return out;
    }    
};

template < typename T > struct Average3x3 : StencilOperator< 3, 3 > {
    template < typename In2DAccessor >
    __device__ T operator()( const In2DAccessor& a ) {
        const real_t W = 1.f / 9.f;
        return W *( a( -1, -1 ) + a( -1, 0 ) + a( -1, 1 ) +
                    a(  0, -1 ) + a(  0, 0 ) + a(  0, 1 ) +
                    a(  1, -1 ) + a(  1, 0 ) + a(  1, 1 ) );                
    }
};


template < typename T > struct Init : StencilOperator< 1, 1 > {
    template < typename InOut2DAccessor >
    __device__ T operator()( const InOut2DAccessor&  ) {
        return ( blockIdx.x + blockIdx.y ) % 2 == 0 ? T( 1 ) : T( 0 );                
    }    
};  

//================================ Stencil  ====================================

//------------------------------------------------------------------------------
template< typename InOutAccessor,
          typename StencilOperator > 
__global__ void apply_stencil_1( InOutAccessor io, StencilOperator op ) { 
    io() = op( io );
}


template< typename InAccessor,
          typename OutAccessor,
          typename StencilOperator > 
__global__ void apply_stencil_2( InAccessor in, OutAccessor out, StencilOperator op ) { 
    out() = op( in );
}

//------------------------------------------------------------------------------
__global__ void apply_3x3average( const real_t* vin, real_t* vout ) {
    real_t out = 0.f;
    const real_t W = 1.f / 9.f;
    const int gridWidth  = gridDim.x * blockDim.x;
    const int gridHeight = gridDim.y * blockDim.y;
    const int row    = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    for( int i = -1; i <= 1; ++i ) {
        int rowIdx = row + i;
#ifndef NO_BOUND_CHECK        
        if( rowIdx < 0 ) rowIdx = 0;
        else if( rowIdx >= gridHeight ) rowIdx = gridHeight - 1;
#endif        
        for( int j = -1; j <= 1; ++j ) {
            int colIdx = column + j;
#ifndef NO_BOUND_CHECK            
            if( colIdx < 0 ) colIdx = 0;
            else if( colIdx >= gridWidth ) colIdx = gridWidth - 1;
#endif            
            out += vin[ rowIdx * gridWidth + colIdx ] * W;
        }        
    }
    vout[ row * gridWidth + column ] = out;
    // note: the loop is faster than explicilt unrolling with multiple calls to get_global_idx_2d    
}

//------------------------------------------------------------------------------
// assumption:
// - GPU grid == core space (i.e. whole grid - halo region)
// - input grid and output grid == whole grid with halo           
__global__ void apply_3x3average_core( const real_t* vin, real_t* vout ) {
    real_t out = 0.f;
    const int inRowOffset = 1; // 2 * ( 3 div 2 ); 3 == stencil height
    const int inColOffset = 1; // 2 * ( 3 div 2 ); 3 == stencil width
    const real_t W = 1.f / 9.f;
    const int gridWidth = gridDim.x * blockDim.x + 2 * inColOffset; // add halo region
    const int row    = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowPlusOffset = row + inRowOffset;
    const int colPlusOffset = column + inColOffset;
    for( int i = -1; i <= 1; ++i ) {
        const int rowStride = ( rowPlusOffset + i ) * gridWidth;     
        for( int j = -1; j <= 1; ++j ) {
            out += vin[ rowStride + colPlusOffset + j ] * W;
        }        
    }
    vout[ rowPlusOffset * gridWidth + colPlusOffset ] = out;
    // note: the loop is faster than explicilt unrolling with multiple calls to get_global_idx_2d    
}


//------------------------------------------------------------------------------
size_t get_global_idx_2d_host( int row, int col, int offRow, int offCol, 
                               int numRows, int numColumns ) {    
    int rowIdx = row + offRow;
    int colIdx = col + offCol;
    rowIdx = std::min( numRows - 1, rowIdx );
    rowIdx = std::max( 0, rowIdx );
    colIdx = std::min( numColumns - 1, colIdx );
    colIdx = std::max( 0, colIdx );      
    return rowIdx * numColumns + colIdx;
}

void apply_3x3average_host_core( const real_t* vin, real_t* vout, 
                                 int num_rows, int num_columns,
                                 int rowOffset, int colOffset,
                                 int inNumColumns ) {
    const real_t W = 1.f / 9.f;
    for( int row = 0; row != num_rows; ++row ) {
        for( int col = 0; col != num_columns; ++col ) {
            real_t out = 0.f;
            for( int i = -1; i <= 1; ++i ) {
                const int rowStride = ( row + rowOffset + i ) * inNumColumns;
                for( int j = -1; j <= 1; ++j ) {
                    const int colIdx = col + colOffset;
                    out += vin[ rowStride + colIdx + j ] * W;
                }        
            }
            vout[ row * num_columns + col ] = out;    
        }
    }                  
}

void apply_3x3average_host( const real_t* vin, real_t* vout, int num_rows, int num_columns ) {
    const real_t W = 1.f / 9.f;
    for( int row = 0; row != num_rows; ++row ) {
        for( int col = 0; col != num_columns; ++col ) {
            real_t out = 0.f;
            for( int i = -1; i <= 1; ++i ) {
                for( int j = -1; j <= 1; ++j ) {
                    out += 
                        vin[ get_global_idx_2d_host( row, col, i, j, num_rows, num_columns ) ] 
                        * W; 
                }        
            }
            vout[ get_global_idx_2d_host( row, col, 0, 0, num_rows, num_columns ) ] = out;    
        }
    }  
                 
}

#define p std::cout << __LINE__ - 1 << "> " << cudaGetErrorString( cudaGetLastError() ) << std::endl;

//==============================================================================
//------------------------------------------------------------------------------
int main( int , char**  ) {
    
    const int NUM_ROWS    = 4096;
    const int NUM_COLUMNS = 4096;
    const int NUM_ELEMENTS = NUM_ROWS * NUM_COLUMNS; 
    const int TOTAL_SIZE = sizeof( real_t ) * NUM_ELEMENTS; // total size in bytes
    // we choose '1' to have the gpu grid match exactly the domain no matter what the
    // grid size is; this allows to avoid bound checks in the GPU code
    const int THREADS_PER_BLOCK_HEIGHT = 1;//16; //number of gpu threads per block along height
    const int THREADS_PER_BLOCK_WIDTH  = 1;//16; //number of gpu threads per block along width
    
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to the number of elements to process
    const int NUMBER_OF_BLOCKS_HEIGHT = 
        ( NUM_ROWS    + THREADS_PER_BLOCK_HEIGHT - 1 ) / THREADS_PER_BLOCK_HEIGHT;
    const int NUMBER_OF_BLOCKS_WIDTH  = 
        ( NUM_COLUMNS + THREADS_PER_BLOCK_WIDTH - 1  ) / THREADS_PER_BLOCK_WIDTH;

    // gpu allocated storage
    real_t* dev_in = 0; //in grid
    real_t* dev_out = 0; //out grid
    cudaMalloc( &dev_in, TOTAL_SIZE );
    cudaMalloc( &dev_out, TOTAL_SIZE );
    
    // reuse Out2DAccessor as I/O accessor for initializing data
    typedef Out2DAccessor< real_t > InOut2DAccessor;

    // init data
    apply_stencil_1<<< dim3( NUM_COLUMNS, NUM_ROWS, 1 ), 1 >>>( InOut2DAccessor( dev_in ),
                                                                Init< real_t >() );
    

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    float elapsed;
    
    // apply averaging kernels

    dim3  blocks( NUMBER_OF_BLOCKS_WIDTH,  NUMBER_OF_BLOCKS_HEIGHT,  1 );
    dim3 threads( THREADS_PER_BLOCK_WIDTH, THREADS_PER_BLOCK_HEIGHT, 1 );

    //==========================================================================
    // 1 - stencil on whole domain
    std::cout << "\n--- NO HALO ---\n" << std::endl; 

    cudaEventRecord( start, 0 );
    apply_stencil_2<<< blocks, threads >>>( In2DAccessor< real_t >( dev_in ),
                                            Out2DAccessor< real_t >( dev_out ),
#ifndef GENERIC_OP                                            
                                            Average3x3< real_t >() );
#else                                            
                                            Average< real_t, 3, 3 >() );
#endif                                            
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );                                                                               
    cudaEventElapsedTime( &elapsed, start, stop );
    // read back result
    std::vector< real_t > vout( NUM_ELEMENTS ); 
    cudaMemcpy( &vout[ 0 ], dev_out, TOTAL_SIZE, cudaMemcpyDeviceToHost );
    // print first and last element of vector
    std::cout << "C++ kernel: time: " << elapsed << " ms - result: " << vout.front() << ".." 
              << vout.back() << std::endl;

    cudaEventRecord( start, 0 );
    apply_3x3average<<< blocks, threads >>>( dev_in, dev_out );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsed, start, stop );
    // read back result
    cudaMemcpy( &vout[ 0 ], dev_out, TOTAL_SIZE, cudaMemcpyDeviceToHost );
    // print first and last element of vector
    std::cout << "C kernel:   time: " << elapsed << " ms - result: " << vout.front() << ".." 
              << vout.back() << std::endl;

    std::vector< real_t > vin( NUM_ELEMENTS );
    cudaMemcpy( &vin[ 0 ], dev_in, TOTAL_SIZE, cudaMemcpyDeviceToHost );
    clock_t begin = clock();
    apply_3x3average_host( &vin[ 0 ], &vout[ 0 ], NUM_ROWS, NUM_COLUMNS );
    clock_t end = clock();
    std::cout << "CPU kernel: time: " << 1000 * ( end - begin ) / double( CLOCKS_PER_SEC ) 
              << " ms - result " << vout.front() << ".." << vout.back() << std::endl;  


    //==========================================================================
    // 2 - stencil on core space (whole grid - halo) only
    std::cout << "\n--- HALO ---\n" << std::endl; 

    typedef Average3x3< real_t > StencilOp;
    
    // the grid uploaded to the GPU is the same as before but we apply the stencil
    // only to an inner subregion of the grid assuming the external part is the halo
    
    const int HALO_WIDTH  = 2 * ( StencilOp::WIDTH  / 2 ); // e.g. 2 * (3 div 2) == 2
    const int HALO_HEIGHT = 2 * ( StencilOp::HEIGHT / 2 ); // e.g. 2 * (3 div 2) == 2
    const int HALO_OFFSET_WIDTH  = HALO_WIDTH  / 2; 
    const int HALO_OFFSET_HEIGHT = HALO_HEIGHT / 2; 
    // block size: the number of threads per block multiplied by the number of blocks
    // must be at least equal to the number of elements to process
    const int CORE_NUMBER_OF_BLOCKS_HEIGHT = 
        ( NUM_ROWS - HALO_HEIGHT + THREADS_PER_BLOCK_HEIGHT  - 1 ) 
        / THREADS_PER_BLOCK_HEIGHT;
    const int CORE_NUMBER_OF_BLOCKS_WIDTH  = 
        ( NUM_COLUMNS - HALO_WIDTH + THREADS_PER_BLOCK_WIDTH - 1 ) 
        / THREADS_PER_BLOCK_WIDTH;

    blocks = dim3( CORE_NUMBER_OF_BLOCKS_WIDTH,  CORE_NUMBER_OF_BLOCKS_HEIGHT, 1 );
    
    std::cout << vin.front() << std::endl;

    cudaEventRecord( start, 0 );
    apply_stencil_2<<< blocks, threads >>>( 
        In2DCoreAccessor< real_t, StencilOp >( dev_in, NUM_COLUMNS ),
        Out2DCoreAccessor< real_t, StencilOp >( dev_out, NUM_COLUMNS ),
        Average3x3< real_t >() );                                            
p
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );                                                                               
p
    cudaEventElapsedTime( &elapsed, start, stop );
    // read back result
    const int SUB_REGION_HEIGHT = NUM_ROWS - HALO_HEIGHT;
    const int SUB_REGION_WIDTH  = NUM_COLUMNS - HALO_WIDTH;
    const int CORE_NUM_ELEMENTS = SUB_REGION_WIDTH * SUB_REGION_HEIGHT;
    
    vout.resize( CORE_NUM_ELEMENTS );

    memcpy_2d_device_to_host( &vout[ 0 ], dev_out, 
                              NUM_ROWS, NUM_COLUMNS,
                              HALO_OFFSET_HEIGHT, HALO_OFFSET_WIDTH,
                              SUB_REGION_HEIGHT, SUB_REGION_WIDTH );
p    
    print2d( &vout[ 0 ], SUB_REGION_WIDTH );
    // print first and last element of vector
    std::cout << "C++ kernel: time: " << elapsed << " ms - result: " << vout.front() << ".." 
              << vout.back() << std::endl;

    cudaEventRecord( start, 0 );
    apply_3x3average_core<<< blocks, threads >>>( dev_in, dev_out );
p
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsed, start, stop );
    // read back result
    memcpy_2d_device_to_host( &vout[ 0 ], dev_out, 
                              NUM_ROWS, NUM_COLUMNS,
                              HALO_OFFSET_HEIGHT, HALO_OFFSET_WIDTH,
                              SUB_REGION_HEIGHT, SUB_REGION_WIDTH );
p
    print2d( &vout[ 0 ], SUB_REGION_WIDTH );
    // print first and last element of vector
    std::cout << "C kernel:   time: " << elapsed << " ms - result: " << vout.front() << ".." 
              << vout.back() << std::endl;

    cudaMemcpy( &vin[ 0 ], dev_in, TOTAL_SIZE, cudaMemcpyDeviceToHost );
p
    begin = clock();
    apply_3x3average_host_core( &vin[ 0 ], &vout[ 0 ], SUB_REGION_HEIGHT, SUB_REGION_WIDTH,
                                HALO_OFFSET_HEIGHT, HALO_OFFSET_WIDTH, NUM_COLUMNS );
    end = clock();
    print2d( &vout[ 0 ], SUB_REGION_WIDTH );
    std::cout << "CPU kernel: time: " << 1000 * ( end - begin ) / double( CLOCKS_PER_SEC ) 
              << " ms - result " << vout.front() << ".." << vout.back() << std::endl;  

    // free memory
    cudaFree( dev_in );
p
    cudaFree( dev_out );
p
    return 0;
}
