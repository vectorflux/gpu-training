/* 
 *   BiCGstab solver based on CUSPARSE and CUBLAS
 *
 *   Read a real (non-complex) sparse matrix from a Matrix Market (v. 2.0) file,
 *   then store it into CUSPARSE COO format.
 *
 *   Usage:  bicgstab.exe [filename] > output
 *
 *       
 *   NOTES:
 *
 *   1) Matrix Market files are always 1-based, i.e. the index of the first
 *      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
 *      OFFSETS ACCORDINGLY offsets accordingly when reading and writing 
 *      to files.
 *
 *   2) ANSI C requires one to use the "l" format modifier when reading
 *      double precision floating point numbers in scanf() and
 *      its variants.  For example, use "%lf", "%lg", or "%le"
 *      when reading doubles, otherwise errors will occur.
 *   3) The NETLIB BiCGStab matlab version is:
%  -- Iterative template routine --
%     Univ. of Tennessee and Oak Ridge National Laboratory
%     October 1, 1993
%     Details of this algorithm are described in "Templates for the
%     Solution of Linear Systems: Building Blocks for Iterative
%     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
%     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
%     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
%
%  [x, error, iter, flag] = bicgstab(A, x, b, M, max_it, tol)
%
% bicgstab.m solves the linear system Ax=b using the 
% BiConjugate Gradient Stabilized Method with preconditioning.
%
% input   A        REAL matrix
%         x        REAL initial guess vector
%         r        REAL right hand side vector
%         M        REAL preconditioner matrix
%         max_it   INTEGER maximum number of iterations
%         tol      REAL error tolerance
%
% output  x        REAL solution vector
%         error    REAL error norm
%         iter     INTEGER number of iterations performed
%         flag     INTEGER: 0 = solution found to tolerance
%                           1 = no convergence given max_it
%                          -1 = breakdown: rho = 0
%                          -2 = breakdown: omega = 0

  iter = 0;                                          % initialization
  flag = 0;

  bnrm2 = norm( r );
  if  ( bnrm2 == 0.0 ), bnrm2 = 1.0; end

  r = r - A*x;   %  r now residual
  error = norm( r ) / bnrm2;
  if ( error < tol ) return, end

  omega  = 1.0;
  r_tld = r;

  for iter = 1:max_it,                              % begin iteration

     rho   = ( r_tld'*r );                          % direction vector
     if ( rho == 0.0 ) break, end

     if ( iter > 1 ),
        beta  = ( rho/rho_1 )*( alpha/omega );
        p = r + beta*( p - omega*v );
     else
        p = r;
     end
 
     p_hat = M \ p;
     v = A*p_hat;
     alpha = rho / ( r_tld'*v );
     s = r - alpha*v;
     if ( norm(s) < tol ),                          % early convergence check
        x = x + alpha*p_hat;
        resid = norm( s ) / bnrm2;
        break;
     end

     s_hat = M \ s;                                 % stabilizer
     t = A*s_hat;
     omega = ( t'*s) / ( t'*t );

     x = x + alpha*p_hat + omega*s_hat;             % update approximation

     r = s - omega*t;
     error = norm( r ) / bnrm2;                     % check convergence
     if ( error <= tol ), break, end

     if ( omega == 0.0 ), break, end
     rho_1 = rho;

  end

  if ( error <= tol ),                              % converged
     flag =  0;
  elseif ( omega == 0.0 ),                          % breakdown
     flag = -2;
  elseif ( rho == 0.0 ),
     flag = -1;
  else                                              % no convergence
     flag = 1;
  end

% END bicgstab.m

 */

#define TOL ((double) 0.0001)
#define MAX_IT 100

#define CLEANUP(s) \
do { \
  printf ("%s\n", s);	      \
  fflush (stdout);					\
} while (0)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas.h"
#include "cusparse.h"
#include "mmio.h"

/* extern int define_solve(int M_, int N_, int nz_,
   double *val_, int *I_, int *J_);  */

int main(int argc, char *argv[])
{
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int M, N, nnz;   
  int i, iter, flag;
  int *cooRowPtrAhost, *cooColPtrAhost, *csrColPtrAhost;
  double *cooValAhost, *bhost, *xhost;
  int *cooRowPtrAdev, *cooColPtrAdev, *csrColPtrAdev;
  double *cooValAdev, *x, *r, *r_tld, *p, *p_hat, *s, *s_hat, *t, *v;
  double bnrm2, snrm2, error, alpha, beta, omega, rho, rho_1, resid;
  
  cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
  cudaError_t cudaStat7,cudaStat8,cudaStat9,cudaStat10,cudaStat11,cudaStat12,cudaStat13;

  cublasStatus cublas_status;
  cusparseStatus_t status;
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descra=0;

  if (argc < 2)
    {
      fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
      exit(1);
    }
  else    
    { 
      if ((f = fopen(argv[1], "r")) == NULL) 
	exit(1);
    }

  if (mm_read_banner(f, &matcode) != 0)
    {
      printf("Could not process Matrix Market banner.\n");
      exit(1);
    }


  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nnz)) !=0)
    exit(1);

  /* initialize CUBLAS */
  cublas_status = cublasInit();
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }


  /* reseve memory for matrices */

  cooRowPtrAhost = (int *) malloc(nnz * sizeof(cooRowPtrAhost[0]));
  cooColPtrAhost = (int *) malloc(nnz * sizeof(cooColPtrAhost[0]));
  csrColPtrAhost = (int *) malloc((N+1) * sizeof(csrColPtrAhost[0]));
  cooValAhost    = (double *) malloc(nnz * sizeof(cooValAhost[0]));
  bhost          = (double *) malloc(N * sizeof(bhost[0]));
  xhost          = (double *) malloc(N * sizeof(xhost[0]));
  for (i=0; i<N; i++){bhost[i]  = 1.0; xhost[i] = 0.0;}


  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (i=0; i<nnz; i++)
    {
      fscanf(f, "%d %d %lg\n", &cooRowPtrAhost[i], &cooColPtrAhost[i], &cooValAhost[i]);
      cooRowPtrAhost[i]--;  /* adjust from 1-based to 0-based */
      cooColPtrAhost[i]--;
    }

  printf("Read matrix %d rows, %d cols, %d nonzeros \n",M,N,nnz);
  for (i=0; i<nnz; i++){
    printf("cooRowIndexHostPtr[%d]=%d ",i,cooRowPtrAhost[i]);
    printf("cooColIndexHostPtr[%d]=%d ",i,cooColPtrAhost[i]);
    printf("cooValHostPtr[%d]=%f \n",i,cooValAhost[i]);
  }

  if (f !=stdin) fclose(f);

  /* allocate GPU memory and copy the matrix and vectors into it */
  cudaStat1 = cudaMalloc((void**)&cooRowPtrAdev,nnz*sizeof(cooRowPtrAdev[0]));
  cudaStat2 = cudaMalloc((void**)&cooColPtrAdev,nnz*sizeof(cooColPtrAdev[0]));
  cudaStat3 = cudaMalloc((void**)&cooValAdev, nnz*sizeof(cooValAdev[0]));
  cudaStat4 = cudaMalloc((void**)&r, N*sizeof(r[0]));
  cudaStat5 = cudaMalloc((void**)&s, N*sizeof(s[0]));
  cudaStat6 = cudaMalloc((void**)&t, N*sizeof(t[0]));
  cudaStat7 = cudaMalloc((void**)&r_tld, N*sizeof(r_tld[0]));
  cudaStat8 = cudaMalloc((void**)&p, N*sizeof(p[0]));
  cudaStat9 = cudaMalloc((void**)&p_hat, N*sizeof(p_hat[0]));
  cudaStat10= cudaMalloc((void**)&s_hat, N*sizeof(s_hat[0]));
  cudaStat11= cudaMalloc((void**)&v, N*sizeof(v[0]));
  cudaStat12= cudaMalloc((void**)&x, N*sizeof(x[0]));
  cudaStat13= cudaMalloc((void**)&csrColPtrAdev,(N+1)*sizeof(csrColPtrAdev[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess) ||
      (cudaStat6 != cudaSuccess) ||
      (cudaStat7 != cudaSuccess) ||
      (cudaStat8 != cudaSuccess) ||
      (cudaStat9 != cudaSuccess) ||
      (cudaStat10!= cudaSuccess) ||
      (cudaStat11!= cudaSuccess) ||
      (cudaStat12!= cudaSuccess)) {
    CLEANUP("Device malloc failed");
    return EXIT_FAILURE;
  }

  cudaStat1 = cudaMemcpy(cooRowPtrAdev, cooRowPtrAhost,
			 (size_t)(nnz*sizeof(cooRowPtrAdev[0])),
			 cudaMemcpyHostToDevice);
  cudaStat2 = cudaMemcpy(cooColPtrAdev, cooColPtrAhost,
			 (size_t)(nnz*sizeof(cooColPtrAdev[0])),
			 cudaMemcpyHostToDevice);
  cudaStat3 = cudaMemcpy(cooValAdev, cooValAhost,
			 (size_t)(nnz*sizeof(cooValAdev[0])),
			 cudaMemcpyHostToDevice);
  cudaStat4 = cudaMemcpy(r, bhost,
			 (size_t)(N*sizeof(r[0])),
			 cudaMemcpyHostToDevice);
  cudaStat5 = cudaMemcpy(x, xhost,
			 (size_t)(N*sizeof(x[0])),
			 cudaMemcpyHostToDevice);

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess)) {
    CLEANUP("Memcpy from Host to Device failed");
    return EXIT_FAILURE;
  }

  /* initialize cusparse library */
  status= cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CUSPARSE Library initialization failed");
    return EXIT_FAILURE;
  }
  /* create and setup matrix descriptor */
  status= cusparseCreateMatDescr(&descra);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("Matrix descriptor initialization failed");
    return EXIT_FAILURE;
  }
  cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);

  status= cusparseXcoo2csr(handle,cooColPtrAdev,nnz,N,
			   csrColPtrAdev,CUSPARSE_INDEX_BASE_ZERO);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("Conversion from COO to CSR format failed");
    return EXIT_FAILURE;
  }

  cudaStat1 = cudaMemcpy(csrColPtrAhost, csrColPtrAdev,
			 (size_t)((N+1)*sizeof(csrColPtrAhost[0])),
			 cudaMemcpyDeviceToHost);
  if ((cudaStat1 != cudaSuccess)) {
    CLEANUP("Memcpy from Device to Host failed");
    return EXIT_FAILURE;
  }

  for (i=0;i<N+1;++i) printf("csr[%d] = %d \n", i,csrColPtrAhost[i]); 

  bnrm2 = cublasDnrm2 (N, r, 1);

  printf("bnrm2 = %e \n", bnrm2);

  status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_TRANSPOSE, N, N, -1.0,
			 descra, cooValAdev, csrColPtrAdev, cooRowPtrAdev, x,
			 1.0, r);   /* r = r - A*x; r is now residual */


  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("Matrix‐vector multiplication failed");
    return EXIT_FAILURE;
  }
  
  error = cublasDnrm2 (N, r, 1) / bnrm2;   /* norm_r = norm(b) */
  if ( error < TOL ) return 0;   /* x is close enough already */

  omega = 1.0;

  cudaStat1 = cudaMemcpy(r_tld, r, (size_t)(N*sizeof(r[0])),
			 cudaMemcpyDeviceToDevice);    /* r_tld = r */
  if ((cudaStat1 != cudaSuccess)) {
    CLEANUP("Memcpy from r to r_tld failed");
    return EXIT_FAILURE;
  }

  for (iter = 0; iter < MAX_IT; ++iter)
    {
      rho = cublasDdot (N, r_tld, 1, r, 1);  /* rho = r_tld'*r */
      printf("rho = %e \n", rho);

      if ( rho == 0.0 ) break;
      
      if ( iter > 0 )
	{
	  beta = (rho/rho_1) * (alpha/omega);
	  cublasDaxpy (N, -omega, v, 1, p, 1);
	  cublasDaxpy (N, 1.0/beta, r, 1, p, 1);
	  cublasDscal (N, beta, p, 1);         /* p = r + beta*( p - omega*v ) */
	}
      else
	{
	  cudaStat1 = cudaMemcpy(p, r,
				 (size_t)(N*sizeof(r[0])),
				 cudaMemcpyDeviceToDevice);    /* p = r */
	  if ((cudaStat1 != cudaSuccess)) {
	    CLEANUP("Memcpy from r to p failed");
	    return EXIT_FAILURE;
	  }
	}

      /* p_hat = M \ p    --> here assume M = I, the identity matrix */

      cudaStat1 = cudaMemcpy(p_hat, p, (size_t)(N*sizeof(x[0])),
			     cudaMemcpyDeviceToDevice);    /* p_hat = p */
      if ((cudaStat1 != cudaSuccess)) {
	CLEANUP("Memcpy from p to p_hat failed");
	return EXIT_FAILURE;
      }

      status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_TRANSPOSE, N, N, 1.0,
			     descra, cooValAdev, csrColPtrAdev, cooRowPtrAdev, p_hat,
			     0.0, v);   /* v = A*p_hat */

      alpha = rho / cublasDdot (N, r_tld, 1, v, 1);  /* alph = rho / ( r_tld'*v ) */
  
      printf("alpha = %e \n", alpha);

      cudaStat1 = cudaMemcpy(s, r, (size_t)(N*sizeof(r[0])),
			     cudaMemcpyDeviceToDevice);    /* s = r */
      if ((cudaStat1 != cudaSuccess)) {
	CLEANUP("Memcpy from r to s failed");
	return EXIT_FAILURE;
      }
      cublasDaxpy (N, -alpha, v, 1, s, 1);
      snrm2 = cublasDnrm2( N, s, 1);


      printf("snrm2 = %e \n", snrm2);
      if ( snrm2 < TOL )
	{
	  cublasDaxpy (N, alpha, p_hat, 1, s, 1);
	  resid = snrm2 / bnrm2;
	  break;
	}
      
      /* s_hat = M \ s    --> here assume M = I, the identity matrix */

      cudaStat1 = cudaMemcpy(s_hat, s, (size_t)(N*sizeof(s[0])),
			     cudaMemcpyDeviceToDevice);    /* s_hat = s */
      if ((cudaStat1 != cudaSuccess)) {
	CLEANUP("Memcpy from s to s_hat failed");
	return EXIT_FAILURE;
      }
  
      status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_TRANSPOSE, N, N, 1.0,
			     descra, cooValAdev, csrColPtrAdev, cooRowPtrAdev, s_hat,
			     0.0, t);                               /* t = A*s_hat */

      omega = cublasDdot(N, t, 1, s, 1)/cublasDdot(N, t, 1, t, 1);  /* omega = ( t'*s) / ( t'*t ) */

      printf("omega = %e \n", omega);

      cublasDaxpy (N, alpha, p_hat, 1, x, 1);
      cublasDaxpy (N, omega, s_hat, 1, x, 1);  /*  x = x + alph*p_hat + omega*s_hat */
  
      cublasDaxpy (N, -omega, t, 1, s, 1);     
      cudaStat1 = cudaMemcpy(r, s, (size_t)(N*sizeof(r[0])),
			     cudaMemcpyDeviceToDevice);       /* r = s - omega*t */
      if ((cudaStat1 != cudaSuccess)) {
	CLEANUP("Memcpy from s to r failed");
	return EXIT_FAILURE;
      }
  
      error = cublasDnrm2( N, r, 1) / bnrm2;
      if ( snrm2 <= TOL ) break;
      if ( omega == 0.0 ) break;
      rho_1 = rho;

      printf("rho_1 = %e \n", rho_1);
    }

  if ( error <= TOL ) {
    flag = 0;
  }
  else if ( omega == 0.0 ) {
    flag = -2;
  }
  else if ( rho == 0.0 ) {
    flag = -1;
  }
  else
    flag = 1;

  if ( !flag ) 
    {
      printf("BiCGStab produced answer with resid %e in %d iterations \n",resid, iter);
    }
  else
    {
      printf("BiCGStab produced error %d after %d iterations \n",flag, iter);
    }
  
  /* shutdown CUBLAS */
  cublas_status = cublasShutdown();
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  return 0;
}

