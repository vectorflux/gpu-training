CUBLAS Exercise: CPU/GPU-split DGEMM

Author: Gilles Fourestey

Goal: Split a computational kernel over CPUs and GPUs to improve performance

Rationale:

GPUs generally are considered as an accelerator attached to a compute node, which may contain a number of processors (or sockets), each with multiple cores. For example, Eiger has 2 sockets each with 6 cores. The common GPU programming model is to let the GPU perform the computationally intensive part of the computation. Even though the GPU excels at computation, often the cores are left idle waiting for the output data to return from the device.

In this exercise we illustrate that at least some kernels can be sensibly distributed over the CPUs and GPUs. This is illustrated with the ubiquitous double-precision matrix-matrix multiplication, also referred to as DGEMM in BLAS terminology. The concept is simple: columns of the matrix B are distributed over CPUs and GPUs as depicted in the following figure:

File:gemm_split.jpg

There are two potential benefits to this scheme: in the first place, less data must be copied between host and device. Secondly the cores can be occupied with sensible work. Clearly this scheme can only work well if the distribution is balanced such that all CPUs and GPUs complete their calculations at nearly the same time, and herein lies the crux of this exercise.

Code: This exercise is split over the following files. You can either download them individually (below) or copy them from the directory indicated by the instructor.

    t3.c -- main program which distributes the mat-mat mult. over CPUs and GPUs 

    cpu_dgemm.c -- wrapper to dgemm routines 

    cpu_dgemm.h -- dgemm interfaces 

    timer.c -- timing routines 

    timer.h -- timing interfaces 

    utils.h -- utilities 

    makefile 

    make.inc 

or directly from the repository branch:

 svn co svn://scm.hpcforge.org/var/lib/gforge/chroot/scmrepos/svn/gpu-training/branches/basel-gpu-workshop-gpgpu_blas

Assignment:

    As is, the code DOES NOT COMPILE. Before calling the cublasdgemm in t3.c, the device instances of matrices A, B, and C must be allocated with cublasAlloc and these instances set using cublasSetMatrix. Currently this is only done for matrix B. Consider the above figure carefully, before filling in the calls for matrices A and C. 

    After this fix, the code should compile. It can then be run either within an interactive session (as explained in yesterday's exercises) or with a batch job. The following can be used as a template: 

 runit -- batch script to run under PBS on eiger with one 12-core node

You can manipulate several environment variables in this file, the most important one being:

 export OMP_NUM_THREADS=6

which specifies the number of threads used. Perform several runs with different problem sizes (the argument passed to the executable, e.g., "./t3 10000").

    The code performs poorly: the best configuration is OMP_NUM_THREADS=1, where all the work is on the GPU and a bit on thread 0. The problem is related to the work distribution algorithm, which takes requires an estimate of the effective peak performance of the GPU. The value: 

#ifdef USEGPU
       gpu_gflops = 50;    /* Determine a sensible value for the max gpu Gflop/s */
#endif

The value of 50 GFlop/s is not correct for any of the GPUs on the system. Using the system attributes returned by the query function,

cudaGetDeviceProperties(&properties, device);

determine a sensible estimate for the peak performance (hint: each GPU "processor" consists of 8 to 32 SIMD units, each capable of performing 2 floating point operation per clock cycle). Recompile and rerun the code. How does performance vary as a function of OMP_NUM_THREADS?

    The "peak" performance suggested above is based on unrealistic assumptions and vendor optimism. Try to think of an alternative way to calculate the effective peak performance (hint: the most realistic measure is the DGEMM performance as N becomes very large). Insert this new peak into the code. You should be able to attain in the realm of 330 GFlop/s on the Fermi card. 

    Note that CPU thread 0 is responsible for "babysitting" the GPU, with which there is certain overhead associated. Thread 0 should thus get less work than other CPU threads. The weighting THREAD_0_FACTOR tries to take this into account: 

#define THREAD_0_FACTOR 0.50

Further tune this parameter to attain the highest possible performance. 
