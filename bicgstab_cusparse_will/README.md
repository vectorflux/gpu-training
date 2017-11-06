'''Author:''' William Sawyer

'''Goal:''' Implement the preconditioned BiCGStab solver using CUSPARSE and CUBLAS library calls.

'''Rationale:'''

GPUs are known to perform well for dense linear algebra -- the DGEMM operation illustrates this well.  For sparse linear algebra their programming is more complex and their performance is more nebulous.  One solution is to use the CUSPARSE library, which implements (a limited number of) sparse operators.  In this exercise we implement the well-known preconditioned BiCGStab iterative solver, which requires only sparse matrix times dense vector multiplications (provided by CUSPARSE), and dense vector operations (provided by CUBLAS).  

The [http://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method Bi-Conjugate Gradient Stabilized Method] is one of many extensions for non-symmetric systems in the 1990's of the famous [http://en.wikipedia.org/wiki/Conjugate_gradient_method Conjugate Gradient (CG) Method] proposed by [http://nvl.nist.gov/pub/nistpubs/jres/049/6/V49.N06.A08.pdf Hestenes and Stiefel] in 1952.  It has several advantages over other [http://en.wikipedia.org/wiki/Iterative_method Iterative Methods], in particular other
[http://en.wikipedia.org/wiki/Iterative_method#Krylov_subspace_methods Krylov subspace methods] in that it:

* Has relatively small memory requirements (unlike the [http://en.wikipedia.org/wiki/Generalized_minimal_residual_method Generalized Minimal Residual Method])

* Requires the matrix-vector multiplication but not the matrix-transposed time vector operation (unlike [http://en.wikipedia.org/wiki/Biconjugate_gradient_method Biconjugate Gradient Method], among others)

* Often has more stable convergence than other solvers

One of the best references for these methods is the [http://www.netlib.org/linalg/html_templates/Templates.html Templates] book on-line.

The BiCGstab algorithm can be formatted succinctly in Matlab: [https://hpcforge.org/scm/viewvc.php/trunk/bicgstab_cusparse_will/bicgstab.m?root=gpu-training&view=markup bicgstab.m]  It is quickly apparent that the iterations consists only of two matrix-vector multiplications and a number of vector operations, and these are to be implemented by CUSPARSE and CUBLAS calls, respectively.

However, the solver in itself does not is not sufficient.
The initial enthusiasm over the CG method was soon dampened by the realization that Krylov subspace methods only converge quickly if the underlying matrix is [http://en.wikipedia.org/wiki/Condition_number well-conditioned].  This is almost never the case in real applications.  However, this deficiency can be relaxed by the use of a [http://en.wikipedia.org/wiki/Preconditioner preconditioner], which is in effect an additional matrix which is a good approximation of the inverse of the matrix, with small computational cost.

This exercise starts with the simplest possible preconditioner, namely the identity matrix (equivalent to applying no preconditioner at all), and incrementally improves upon this.

'''Code:'''

* [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/bicgstab.c?root=gpu-training&view=markup bicgstab.c -- bicgstab routine and main program]

* [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/mmio.c?root=gpu-training&view=markup mmio.c -- utilities to read/write sparse matrices]

* [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/mmio.h?root=gpu-training&view=markup mmio.h -- interfaces to sparse matrix I/O ]

* [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/Makefile?root=gpu-training&view=markup Makefile]

Or with SVN:

  svn co svn://scm.hpcforge.org/var/lib/gforge/chroot/scmrepos/svn/gpu-training/branches/basel-gpu-workshop-bicgstab

The key CUSPARSE functionality employed is (refer to CUSPARSE lecture material):

* Create a handle to the sparse system:

  status= cusparseCreate(&handle);

* Create a sparse matrix descriptor:

  status= cusparseCreateMatDescr(&descra);

* Define the attributes of the sparse matrix (e.g., matrix type, zero- or one-based indexing):

  cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);

* Conversion of the matrix from coordinate (COO) format (native to the MTX files) to compressed sparse row (CSR) format, the only one support in the matrix-vector multiply.

  status= cusparseXcoo2csr(...);

* The matrix-vector multiplication:

  status= cusparseDcsrmv(...);


'''Notes:'''

* The most effective way to develop linear algebra software is to compare with working Matlab components.  Teaching Matlab is well beyond the scope of this course, but if you have Matlab experience, you can load the environment with

  module load matlab
  matlab -nodesktop

* Download the following Matlab files:

  [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/bicgstab.m?root=gpu-training&view=markup bicgstab.m -- BiCGstab function]

  [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/mminfo.m?root=gpu-training&view=markup mmio.m -- Get information from sparse matrix MTX file]

  [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/mmread.m?root=gpu-training&view=markup mmread.m -- Read a sparse matrix MTX file]

  [https://hpcforge.org/scm/viewvc.php/branches/basel-gpu-workshop-bicgstab/mmwrite.m?root=gpu-training&view=markup mmwrite.m -- Write a sparse matrix MTX file]

* After storing these in the current directory, you can get help in the Matlab console with, e.g.,

  matlab> help bicgstab

'''Assignment:'''

* The code is compiled with the NVIDIA C/C++ compiler; to load it:

  module load cuda

A Makefile is availabe to compile the code.

* The code "does not compile"!   Look for the comments labeled "ASSIGNMENT".  First one must make additions to the matrix COO to CSR conversion "cusparseXcoo2csr".  Here one must be careful, for this routines assumes that the coordinate indices are ordered.  It turns out the MTX file only the column indices are monotonically increasing.  In standard "cusparseXcoo2csr" examples, the row indices are assumed to be in increasing order.  A few lines later, complete the argument list of "cudaMemcpy" to move the CSR index array to the device

* Complete the argument list for the three calls to the matrix vector multiply "cusparseDcsrmv" (one outside the iteration, two inside).  Note that all arguments except the vectors remain the same in these three calls.

* After successfully compiled, the code is run on a GPU (in an interactive session a GPU node) with:

  ./bicgstab.exe matrix_market_filename.mtx

Use the test matrix defined specifically for this exercise:

  ./bicgstab.exe /project/csstaff/inputs/Matrices/MatrixMarket/SMALL/e05r0000_dd.mtx

The residual should converge to an tolerance "TOL" of less than 10e-4 in a small number of iteration steps.  Now try the closely related linear system:

  ./bicgstab.exe /project/csstaff/inputs/Matrices/MatrixMarket/SMALL/e05r0000.mtx

This does not converge, even if "MAX_IT" is substantially increased from the default value of 100.  What do you suppose is the key difference between these two matrices (Matlab users should know how to verify the suspicion)?

* Extend the version by reading in a precomputed [http://en.wikipedia.org/wiki/Preconditioner preconditioner].  This is an approximation for the inverse of the matrix.  The application of the preconditioner  is then a matrix-vector multiplication. Test this on the "e05r0000.mtx" matrix.
