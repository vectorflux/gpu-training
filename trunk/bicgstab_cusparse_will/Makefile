INC_DIR := /apps/eiger/Cuda-3.2/cuda/include 
LIB_DIR := /apps/eiger/Cuda-3.2/cuda/lib64
LIBS    := -lcusparse -lcublas -lcuda

#
# make all builds both the GPU and host versions versions
#

all:	cusparse_test.exe bicgstab.exe

%.o: %.c
	gcc -c $< -I. -I$(INC_DIR) -O3

cusparse_test.exe: cusparse_test.o
	nvcc -o cusparse_test.exe cusparse_test.o -O3 -L$(LIB_DIR) $(LIBS)

bicgstab.exe: bicgstab.o mmio.o
	nvcc -o bicgstab.exe bicgstab.o mmio.o -O3 -L$(LIB_DIR) $(LIBS)

clean:
	rm -f *.o *.exe *.mod *.gpu *.ptx *.bin *.s

distclean: clean
	rm -f *~
