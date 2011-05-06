#include <stdio.h>       /* standard I/O routines                 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>       /* standard I/O routines                 */
#include <omp.h>

#include <sys/time.h>
//#include <cblas.h>

#define USEGPU 

#ifdef USEGPU
#include <cuda.h>
#include <math.h>
#include <cublas.h>
#include <cuda_runtime_api.h>
#endif
//#include <magma.h>

#include "utils.h"
#include "cpu_dgemm.h"
#include "timer.h"

#ifdef  USEGPU
#define NUM_THREADS_GPU 1
#define NUM_THREADS_GPU 0
#endif

#define NN 15000;




int main (int argc, char *argv[])
{

#ifdef USEGPU
	cublasInit( );
#endif
        int N = NN;
        if ( argc == 2 )
        {
                N = atoi(argv[1]);
        }
        int M = N;
	int K = N;


#ifdef USEGPU
	//    printout_devices( );
	//    printf("\n");
	double gpu_frequency;
	double gpu_numProcessors;

	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) 
	{

		int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++) 
		{
			struct cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			printf("\nThe Properties of the Device with ID %d are\n",device);
			printf("\tDevice Name : %s",properties.name);
			printf("\n\tDevice Memory Size (in bytes) : %u",properties.totalGlobalMem);
			printf("\n\tDevice frequency : %d",properties.clockRate);
			gpu_frequency = properties.clockRate/1.e6;
			printf("\n\tDevice processor count : %d\n\n",properties.multiProcessorCount);
			gpu_numProcessors = properties.multiProcessorCount;

			if (max_multiprocessors < properties.multiProcessorCount) 
			{
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		//cudaSetDevice(max_device);
	}
#endif


    int nthreads, tid, procs, maxthreads, inpar, dynamic, nested;

    /* Get environment information */
    procs      = omp_get_num_procs();
    nthreads   = omp_get_num_threads();
    maxthreads = omp_get_max_threads();
    inpar      = omp_in_parallel();
    dynamic    = omp_get_dynamic();
    nested     = omp_get_nested();


	char* omp_num_threads = getenv("OMP_NUM_THREADS");

	printf("OMP_NUM_THREADS = %s\n", omp_num_threads);
	double t_gflops = 0.;
	double cpu_gflops = 0.;
	double gpu_gflops = 0.; 

#ifdef USEGPU
	gpu_gflops = 300;
#endif
	cpu_gflops += maxthreads*4*2.2;

	t_gflops = gpu_gflops + cpu_gflops;

	int Ncpu = 0;
	int Ngpu = 0;

	Ngpu    = (int) N*gpu_gflops/(cpu_gflops + gpu_gflops);
	Ncpu    = (int) N - Ngpu;

	printf("Ncpu = %d, Ngpu = %d\n", Ncpu, Ngpu);
	//pthread_t  threads[NUM_THREADS_GPU + NUM_THREADS_CPU];

	double  alpha =  1.;
	double  beta  = -1.;

	// int N1  = N/NUM_THREADS;
	// int N2  = N - N1;

	int lda    = M;
	int ldb    = K;
	int ldc    = M;

	int size_A = K*lda;
	int size_B = N*ldb;
	int size_C = N*ldc;

	double *A  = (double*) malloc(sizeof(double)*size_A);
	if (A == 0) printf("Could not allocate A.\n");

	double *B  = (double*) malloc(sizeof(double)*size_B);
	if (B == 0) printf("Could not allocate B.\n");

	double *C  = (double*) malloc(sizeof(double)*size_C);
	if (C == 0) printf("Could not allocate C.\n");

	double *Cg = (double*) malloc(sizeof(double)*size_C);
	if (Cg == 0) printf("Could not allocate Cg.\n");

	fill(A,  size_A,  31.);
	eye (B,     ldb,   N );
	fill(C,  size_C,  31.);
	fill(Cg, size_C,   0.);


	int t;

	char transa = 'n';
	char transb = 'n';

#if 1
	/* Print environment information */
	printf("Number of processors          = %d\n", procs);
	printf("Number of threads             = %d\n", nthreads);
	printf("Max threads                   = %d\n", maxthreads);
	printf("In parallel?                  = %d\n", inpar);
	printf("Dynamic threads enabled?      = %d\n", dynamic);
	printf("Nested parallelism supported? = %d\n", nested);
#endif

	int nCPU[nthreads + 1];

	nCPU[0]          = 0;
	nCPU[maxthreads] = Ncpu;

	int ii; 
	int temp = Ncpu;
	int mt   = maxthreads;

	for (ii = 1; ii < maxthreads; ++ii)
	{
		if ((ii == 1) && maxthreads > 1) 
		{
			nCPU[1] = 0.90*Ncpu/maxthreads;
			temp = Ncpu - nCPU[1];
			mt--;
		}	
		else nCPU[ii] = nCPU[ii - 1] + temp/mt;
	}
	for (ii = 0; ii < maxthreads + 1; ++ii)
		printf("%d ", nCPU[ii]);

	printf("GPU : %d\n", N);

	struct timeval otime;

	beginTimer(&otime);

#pragma omp parallel shared(nthreads) private(tid)
	{
		tid = omp_get_thread_num();
		struct timeval time;

		nthreads = omp_get_num_threads();
#pragma omp master
		{
			printf("Number of threads = %d\n", nthreads);
		}

		printf("Thread %d is starting ... \n", tid);

		//	cpu_dgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); 

		printf("ok\n");

		beginTimer(&time);

		double* d_A_m;
		double* d_B_m;
		double* d_C_m;

		//timer.start();

		int size_A_m = K   *lda;
		int size_B_m = Ngpu*ldb;
		int size_C_m = Ngpu*ldc;



#pragma omp master
		{
#ifdef USEGPU
			struct timeval gputime;
			cublasStatus status;

			int nGPU[NUM_THREADS_GPU + 1];

			status = cublasAlloc( size_A_m, sizeof(double), (void**)&d_A_m );
			if (status) printf("status error %d\n", status);
			status = cublasAlloc( size_B_m, sizeof(double), (void**)&d_B_m ) ;
			if (status) printf("status error %d\n", status);
			status = cublasAlloc( size_C_m, sizeof(double), (void**)&d_C_m ) ;
			if (status) printf("status error %d\n", status);

			beginTimer(&gputime);
			status = cublasSetMatrix( M, K   , sizeof( double ), A           , lda, d_A_m, lda ) ;
			if (status) printf("status error %d\n", status);
			status = cublasSetMatrix( K, Ngpu, sizeof( double ), B + Ncpu*ldb, ldb, d_B_m, ldb ) ;
			if (status) printf("status error %d\n", status);
			status = cublasSetMatrix( M, Ngpu, sizeof( double ), C + Ncpu*ldc, ldc, d_C_m, ldc ) ;
			if (status) printf("status error %d\n", status);

			double Gtime = endTimer(&gputime);
			printf("GPU : %g Bandwidth GB/s H->D (pinned memory) for %d (%g)\n", sizeof(double)*(size_A_m+size_B_m+size_C_m)/1.0e6/Gtime, Ngpu, Gtime);
			printf("Running cublas ...");

			cublasDgemm('n',
					'n',
					M,
					Ngpu,
					K,
					alpha,
					d_A_m,
					lda,
					d_B_m,
					ldb,
					beta,
					d_C_m,
					ldc);
	//		status = cublasGetMatrix( M, Ngpu, sizeof( double ), d_C_m, ldc, C + Ncpu*ldc, ldc ) ;
//			                         if (status) printf("status error %d\n", status);


			printf("ok\n");


			status = cublasGetError();
			if (status) printf("status error %d\n", status);

	//		status = cublasGetMatrix( M, Ngpu, sizeof( double ), d_C_m, ldc, C + Ncpu*ldc, ldc ) ;
	//		if (status) printf("status error %d\n", status);
#endif
		}


		cpu_dgemm(transa, transb, 
				M, nCPU[tid + 1] - nCPU[tid], K, 
				alpha, 
				A,                 lda, 
				B + ldb*nCPU[tid], ldb, 
				beta, 
				C + ldc*nCPU[tid], ldc); 

		double Etime = endTimer(&time);

#pragma omp master
                {
#ifdef USEGPU
			 cublasStatus status = cublasGetMatrix( M, Ngpu, sizeof( double ), d_C_m, ldc, C + Ncpu*ldc, ldc ) ;
                         if (status) printf("status error %d\n", status);
#endif	
		}

		if (tid == 0) printf("Thread %d: %g Gflops (%g)\n", tid, 2.*M*((nCPU[tid + 1] - nCPU[tid]) + Ngpu)*K/1e6/Etime, Etime);
		else printf("Thread %d: %g Gflops (%g)\n", tid, 2.*M*(nCPU[tid + 1] - nCPU[tid])*K/1e6/Etime, Etime);


	} // thread disband


	printf("Overall Gflops = %f %f percent of peak\n", 2.*M*N*K/1e6/endTimer(&otime),  2.*M*N*K/1e6/endTimer(&otime)/t_gflops);      	

	printf("||C - Cg||_max = %f\n", verifyResult(C, Cg, M, N));

	free(A);
	free(B);
	free(C);
	free(Cg);

	exit(0);

}
