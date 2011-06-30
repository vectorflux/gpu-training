// !!!!  WORK IN PROGRESS !!!
// #CSCS CUDA Training 
//
// #Example XX - CUDA + MPI
//
// #Author Ugo Varetto
//
// #Goal: perform concurrent execution on one or more GPUs on the same node and
//        remote nodes 
//
// #Rationale: 
//
// #Solution: 
//          
// #Code: 
//        
// #Compilation: 
//
// #Execution: e.g. mpirun -np 16 -hosts=eiger180,eiger181 ./cudampi
//
// #Note: it is *not* possible to use an MPI compiler wrapper when using nvcc, it is 
//        therefore required to invoke nvcc with the proper set of library/include paths and library;
//        e.g. nvcc -L/apps/eiger/mvapich2/1.6/mvapich2-gnu/lib -lmpich -lpthread -lrdmacm -libverbs \
//             -libumad -I/apps/eiger/mvapich2/1.6/mvapich2-gnu/include cudampi.cu -o cudampi
//        one trick is to compile a C file with the mpi compiler wrapper (mpicc or mpicxx) then
//        issue an 'ldd <executable>' command to have all the libraries and library paths listed,
//        include paths can usually be easily inferred from the library path (replace 'lib' with 'include')

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char** argv ) {
    int rank, size, len;
    char nid[MPI_MAX_PROCESSOR_NAME];

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Get_processor_name( nid, &len );
    printf( "Hello world from process %d of %d -- Node ID = %s\n", rank, size, nid );
    MPI_Finalize();
    return 0;
}

