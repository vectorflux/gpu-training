void dgemm_ (char *transa,
             char *transb,
             int *m, int *n, int *k,
             double *alpha, double *A, int *lda,
             double *B, int *ldb,
             double *beta, double *C, int *ldc);


void* cpu_dgemm(const char  transa, 
				 const char  transb,
				 const int    M, 
				 const int    N,
                 const int    K, 
				 const double alpha, 
				 const double *A,
                 const int    lda, 
				 const double *B, 
				 const int    ldb,
                 const double beta, 
				 double       *C, 
				 const int    ldc) 
{
	dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
