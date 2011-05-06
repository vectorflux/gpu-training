include ./make.inc

all: t3


HEADERS1 = timer.h cpu_dgemm.h utils.h
SOURCES1 = timer.c cpu_dgemm.c t3.c  

OBJ1 = $(SOURCES1:.c=.o)

t3: $(OBJ1)
	$(CC) $(LDOPTS) $(OPTS) $(LIBDIR) $(LIB) $(INC) $(OBJ1) -o t3 

HEADERS2 = timer.h memcopy.h utils.h
SOURCES2 = timer.c  memcopy.c  

OBJ2 = $(SOURCES2:.c=.o)

memcopy: $(OBJ2)
	$(CC) $(LDOPTS) $(OPTS) $(LIBDIR) $(LIB) $(INC) $(OBJ2) -o memcopy 

#.SUFFIXES: .c .cpp .cu .o

%.o : %.c
	echo $(OBJ)
	$(CC) $(OPTS) $(INC) -c $<


clean:
	echo $(OBJ1) $(OBJ2) t3 memcopy 
	rm -f $(OBJ1) $(OBJ2) t3 memcopy 


dist:

	tar cvfz cscs_utils.tgz $(HEADERS1) $(SOURCES1) $(HEADERS2) $(SOURCES2) make.inc makefile
.PHONY: clean

