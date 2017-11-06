Exercise 6: Random-access Streams

Author: John D. McCalpin, Joe R. Zagar, Dorian Krause

Goal: Quantify the effect of non-contiguous memory access patterns on the sustained bandwidth, the benefits of texture memory and (to some extent) effectiveness of the cache hierarchy in GF200 cards.

Rationale: A modified STREAM benchmark is implemented. An addititional array "r" is introduced which can be used to modify the access pattern. Two versions are foreseen: One uses directly indirect accesses, the other one using the texture memory. The latter must be implemented by the course participant. The memory access pattern can be chosen:

    DEFAULT: Stride-one/contiguous
    RAND: Random access pattern
    LOWER: All threads access the same lower segment of the memory 

Compilation:

 nvcc -arch=sm_13 -DV1|V2 -DN=[...] [-DXXX ] ex_randstream.cu -o ex_randstream -lcuda

where XXX is "DEFAULT", "RAND" or "LOWER".

Execution: The best way is to run it in a loop, e.g.,

 for i in V1 V2 
 do 
   nvcc -arch=sm_13 -D${i} [-D..] -DN=$(( 256*65535 )) ex_randstream.cu -lcuda
   ./a.out
 done

One should compare the sustained bandwidths between V1 and V2.

Notes:

    try on both G90 and GF100 architectures to verify the impact of L1 cache 

Code: ex_randstream.cu

Assignment:

    Compile the V1 version for several problem sizes and for the DEFAULT, RAND and LOWER cases, e.g., 

 nvcc -arch=sm_13 -DV1 -DN=16776960 ex_randstream.cu -o ex_randstream
 nvcc -arch=sm_13 -DV1 -DN=16776960 -DRAND ex_randstream.cu -o ex_randstream
 nvcc -arch=sm_13 -DV1 -DN=16776960 -DLOWER ex_randstream.cu -o ex_randstream

on at least two GPU types. Are the performance ratios for DEFAULT/RAND/LOWER constant in the problem size?

    Implement the V2 (texture memory) version. Texture memory is defined as: 

 #ifdef V2
 texture<float, 1, cudaReadModeElementType> texrefA,
                                          texrefB,
                                          texrefC;
 #endif

and these texture memory references are bound to device variables, for example as follows:

     cudaBindTexture( 0, texrefA, da, N*sizeof(float) );

Edit "ex_randstream.cu" in the locations marked by "ASSIGNMENT".

   if(i < N)
       c[i] = a[r[i]];

becomes

 c[i] = tex1Dfetch(texrefA, r[i]);

    Try implementing additional access patterns, for example (without texture), e.g. by adding a shift to the "r" array 

 r[i] = (i+3) % N;

Solution Code: randstream.cu 
