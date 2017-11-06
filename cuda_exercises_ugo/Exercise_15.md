Example 15: Concurrent Kernels

Author: Ugo Varetto

Goal: execute kernels on the GPU in parallel, check serialized versus parallel performance

Rationale: Fermi-based cards allow kernels to run in parallel

Solution:

    run each kernel in a separate stream
    record a global start and stop event before and after execution of the kernel launch loop
    sync stop event
    [optional] have each kernel store its timing information into an array and use a separate kernel to report timing information after all other kernel have been executed - requires stream to wait for the recording of events in the other streams 

Code: 15_concurrent-kernels.cu 
