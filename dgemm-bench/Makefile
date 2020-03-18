
SHELL =  /bin/bash
GCCMODULE = gcc/8.4.0
MKLMODULE =  mkl/2019.5.281
OPENBLASMODULE =  OpenBLAS/0.3.6
CFLAGS = -O3 -std=c99
BENCHMARKS = dgemm_timing dgemm_timing_openblas
default: $(BENCHMARKS)

dgemm_timing_openblas: dgemm_with_timing_openblas.c
	( module load $(GCCMODULE); \
	module load $(OPENBLASMODULE); \
	$(CC) $(CFLAGS) dgemm_with_timing_openblas.c -o dgemm_timing_openblas -I $$OPENBLAS_HOME/include -L $$OPENBLAS_HOME/lib -lopenblas -lpthread ; \
	)

dgemm_timing: dgemm_with_timing.c
	( module load $(GCCMODULE); \
	  module load $(MKLMODULE); \
	$(CC) $(CFLAGS) dgemm_with_timing.c -o dgemm_timing -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm \
	)

bench: $(BENCHMARKS)
	./benchit
	./benchit.openblas

clean:
	- /bin/rm $(BENCHMARKS)
	  
