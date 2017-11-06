Installing MAGMA based on MKL on Eiger

    Download and unpack MAGMA 

 wget http://www.cs.utk.edu/~tomov/magma_1.0.0-rc5.tar.gz
 zcat magma_1.0.0-rc5.tar.gz | tar xf -

    MAGMA must be compiled on a GPU-enabled node: 

 qsub -I -l select=1:ncpus=1:mem=1gb:gpu=fermi -l cput=01:30:00,walltime=01:30:00  -q feed@eiger170

    Prepare for the installation: 

 cd magma_1.0.0-rc5
 module load cuda/3.2 mkl
 # Create make.inc from one of the templates 
 # See Docs for an MKL-based example
 make

    Run tests 

 cd testing
 ./testing_dgemm

    The output might look like this: 


device 0: Tesla M2050, 1147.0 MHz clock, 2687.4 MB memory device 1: Tesla M2050, 1147.0 MHz clock, 2687.4 MB memory

Usage:

 testing_dgemm [-NN|NT|TN|TT] [-N 1024] 


 Testing transA = N  transB = N
     M    N    K     MAGMA GFLop/s    CUBLAS GFlop/s       error
 ==================================================================
  1024  1024  1024       276.74           279.37         0.000000e+00
  1280  1280  1280       290.44           290.75         0.000000e+00
  1600  1600  1600       295.40           295.06         0.000000e+00
  2000  2000  2000       282.47           284.35         0.000000e+00
  2500  2500  2500       283.74           288.39         0.000000e+00
  3125  3125  3125       297.28           272.63         0.000000e+00
  3906  3906  3906       290.97           294.39         0.000000e+00
  4882  4882  4882       295.49           291.81         0.000000e+00
  6102  6102  6102       297.60           293.45         0.000000e+00
  7627  7627  7627       298.28           298.71         0.000000e+00

    Take a look at some individual CUDA implementations: 

 cd ../magmablas
 more dgemm_fermi.cu
 # Stan gave a long discussion on various aspects of the CUDA programming

    Study some high level drivers, e.g., 

 cd ../src
 more dgeqrf_gpu.cpu
 # Stan explained several issues in the CPU/GPU hybrid implementation for the QR factorization

    Get sparse examples, and add the GPU implementation for the Cholesky QR iteration: 

 wget http://www.cs.utk.edu/~tomov/examples.tar.gz
 #edit chol_qr_it.cu
