
SHELL =  /bin/bash
GCCMODULE = gcc/8.3.0
CFLAGS = -O3 -std=c99
BENCHMARKS = dgemm_timing dgemm_timing_openblas
default: $(BENCHMARKS)

dgemm_timing_openblas: dgemm_with_timing_openblas.c
	( module load $(GCCMODULE); \
	module load OpenBLAS; \
	$(CC) $(CFLAGS) dgemm_with_timing_openblas.c -o dgemm_timing_openblas -I $$OPENBLAS_HOME/include -L $$OPENBLAS_HOME/lib -lopenblas -lpthread ; \
	)

dgemm_timing: dgemm_with_timing.c
	( module load gcc/8.3.0; \
	  module load mkl/2019.5.281; \
	  source `which mklvars.sh` intel64; \
	$(CC) $(CFLAGS) dgemm_with_timing.c -o dgemm_timing -I/opt/intel/mkl/include -L/opt/intel/lib -L/opt/intel/mkl/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm \
	)

bench: $(BENCHMARKS)
	./benchit
	./benchit.openblas

clean:
	- /bin/rm $(BENCHMARKS)
