// "mpi + cuda reduction" 

#include <cuda.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include "mpierr.h"


int main( int argc, char** argv ) {

    int numtasks = 0;
    int task     = 0;
    // INIT ENV
    MPI_( MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN ) );
    MPI_( MPI_Init( &argc, &argv ) );
    MPI_( MPI_Comm_size( MPI_COMM_WORLD, &numtasks ) );
    MPI_( MPI_Comm_rank( MPI_COMM_WORLD, &task  ) );
    
    // PER TASK DATA INIT - in the real world this is read from file
    // through the MPI_File... functions or, less likely received from
    // the root process

    // SELECT GPU = task % <num gpus on node>, note that with this
    // approach it is possible to support nodes with a different number of GPUs
    
    // MOVE DATA TO GPU

    // INVOKE KERNEL

    // MOVE DATA TO CPU

    // REDUCE (SUM) ALL ranks -> rank 0

    // IF RANK == 0 -> PRINT RESULT

    // RELEASE GPU RESOURCES

    // RELEASE MPI RESOURCES
 
    
    MPI_( MPI_Finalize() );

    return 0;
}
