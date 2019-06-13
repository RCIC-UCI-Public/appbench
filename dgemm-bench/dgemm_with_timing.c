#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 100

int main()
{
    double *A, *B, *C;
    int m, n, p, i, r;
    double alpha, beta;
    double s_initial, s_elapsed;

    printf ("\n This test measures performance of Intel(R) MKL function dgemm \n"
            " computing real matrix C=A*B, where A, B, and C are n-by-n matrices\n\n");
	printf(" Printing matrix size n vs elapsed time (ms) and gigaflops\n");
	alpha = 1.0; beta = 0.0;

	for(int s = 200; s <= 2000 ; s = s+200){
    	m = s, p = s, n = s;

    	A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    	B = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
    	C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );

    	if (A == NULL || B == NULL || C == NULL) {
        	printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        	mkl_free(A);
        	mkl_free(B);
        	mkl_free(C);
        	return 1;
    	}

    	for (i = 0; i < (m*p); i++) {
        	A[i] = (double)(i+1);
    	}

    	for (i = 0; i < (p*n); i++) {
        	B[i] = (double)(-i-1);
    	}

    	for (i = 0; i < (m*n); i++) {
        	C[i] = 0.0;
    	}

    	// Making the first run of matrix product using Intel(R) MKL dgemm function
        // via CBLAS interface to get stable run time measurements.
    	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                	m, n, p, alpha, A, p, B, n, beta, C, n);

    	s_initial = dsecnd();
    	for (r = 0; r < LOOP_COUNT; r++) {
        	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            	        m, n, p, alpha, A, p, B, n, beta, C, n);
    	}
    	s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

	double gflops = (2.0*s)*s*s/s_elapsed*1e-9;
		
    	printf ("%d\t%.5f\t%.1f\n", s, (s_elapsed * 1000),gflops);//unit: millisecond
    
    	mkl_free(A);
    	mkl_free(B);
    	mkl_free(C);
    
//    	if (s_elapsed < 0.9/LOOP_COUNT) {
//        	s_elapsed=1.0/LOOP_COUNT/s_elapsed;
//        	i=(int)(s_elapsed*LOOP_COUNT)+1;
//        	printf(" It is highly recommended to define LOOP_COUNT for this example on your \n"
//            	   " computer as %i to have total execution time about 1 second for reliability \n"
//            	   " of measurements\n\n", i);
//    	}
	}

    printf (" Test completed. \n\n");
    return 0;
}
