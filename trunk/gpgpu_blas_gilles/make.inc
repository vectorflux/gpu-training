#
	CPU_GFLOPS = 8.8
	GPU_GFLOPS = 300
	GPU_FLAG   = USEGPU

	CC        = cc 
	CPP       = CC
	NVCC      = nvcc
	FORT      = ftn

	ARCH      = ar
	ARCHFLAGS = cr
	RANLIB    = ranlib

	OPTS      = -std=c99 -O2 -D$(GPU_FLAG) -DCPU_GFLOPS=$(CPU_GFLOPS) -DGPU_GFLOPS=$(GPU_GFLOPS) -fopenmp
	NVOPTS    = --compiler-options -fno-strict-aliasing  -arch sm_20 \
                    -DUNIX -O2
	LDOPTS    =  
 
	LIB       = -lcudart -lcublas -lgoto2 -lgfortran

	CUDADIR   = /apps/eiger/Cuda-3.2/cuda
	HOMEDIR   = /users/wsawyer

	LIBDIR   += -L$(CUDADIR)/lib64/ -L/apps/eiger/magma_0.2/lib -L/apps/eiger/gotoblas2/1.13/gnu_432/lib/omp1threads1
	INC       = -I$(CUDADIR)/include/
	
