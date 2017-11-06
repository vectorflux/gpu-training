Exercise 1 - retrieve device info

Author: Ugo Varetto

Goal: compute the maximum size for a 1D grid layout. i.e. the max size for 1D arrays that allows to match a GPU thread with a single array element.

Rationale: CUDA requires client code to configure the domain layout as a 1D or 2D grid of 1,2 or 3D blocks; it is not possible to simply set the GPU layout to match the domain layout as is the case with OpenCL.

Solution: the max size for a 1D memory layout is computed as (max num blocks per grid) x (max num threads per block).

Code: This program finds all attached devices and prints all the available information for each device. The relevant information here is:

    . deviceProp.maxGridSize[0] // max number of blocks in dimension zero
    . deviceProp.maxThreadsDim[0] // max number of threads per block along dimension 0
    . deviceProp.maxThreadsPerBlock // max threads per block
    . (optional) deviceProp.totalGlobalMem //total amount of memory) 

proper code should perform some minimal error checking and iterate over all the available devices.

Compilation: nvcc -arch=sm_13 1_device-query.cu -o 1_device-query

Execution: ./1_device_query

Notes::

    by default the code prints all the information available for each graphics card; #define MINIMAL to have the code print out only the relevant information 

    the code is C++ also because the default compilation mode for CUDA is C++, all functions are named with C++ convention and the syntax is checked by default against C++ grammar rules 

    -arch=sm_13 allows the code to run on every card available on Eiger and possibly even on students' laptops; it's the identifier for the architecture before Fermi (sm_20) 

Assignment:

    Try the existing program out on all the existing devices. Go through the complete output (i.e., compiled without -DMINIMAL) and try to understand each line. Ask the course assistants if some are unclear. Determine the nodes which have access to more than one GPU. 

    At the very end of the code there is a placeholder for the maximum 1D grid layout size. Devise the appropriate expression (the current value, 999999999, is incorrect ;-) to calculate it and extend the program accordingly. 

    Run your revised program on all GPU types. Is the maximum 1D grid layout size always the same? 
