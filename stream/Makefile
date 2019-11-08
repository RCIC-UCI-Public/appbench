SHELL = /bin/bash
ARRAYSIZE=160000000
CC = gcc
CFLAGS = -O3 -funroll-all-loops -march=skylake -static-libgcc -mcmodel=medium -DSTREAM_ARRAY_SIZE=$(ARRAYSIZE)
CFLAGS = -O3 -funroll-all-loops -static-libgcc -mcmodel=medium -DSTREAM_ARRAY_SIZE=$(ARRAYSIZE)

FF = gfortran
FFLAGS = -O3  -DSTREAM_ARRAY_SIZE=$(ARRAYSIZE)
MODULE = gcc/6.5.0

all: stream_c stream_c_omp 

stream_f: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FF) $(FFLAGS) -c stream.f
	$(FF) $(FFLAGS) stream.o mysecond.o -o $@ 

stream_c: stream.c
	module load $(MODULE); \
	$(CC) $(CFLAGS) stream.c -o $@

stream_c_omp: stream.c
	module load $(MODULE); \
	$(CC) $(CFLAGS) -fopenmp stream.c -o $@

bench: stream_c_omp
	module load $(MODULE); ./benchit

clean:
	rm -f stream_f stream_c stream_c_omp *.o

