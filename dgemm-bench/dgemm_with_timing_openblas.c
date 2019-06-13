#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>
#include <time.h>

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 100

int main()
{
    double *A, *B, *C;
    int m, n, p, i, r;
    double alpha, beta;
    double s_elapsed;

    printf ("\n This test measures performance of OpenBLAS function dgemm \n"
            " computing real matrix C=A*B, where A, B, and C are n-by-n matrices\n\n");
	printf(" Printing matrix size n vs elapsed time (ms) and gigaflops\n");
	alpha = 1.0; beta = 0.0;

	// for(int s = 2; s <= 2000 ; s = s+100){
	for(int s = 200; s <= 2000 ; s = s+200){
    	m = s, p = s, n = s;

    	A = (double *)malloc( m*p*sizeof( double ) );
    	B = (double *)malloc( p*n*sizeof( double ) );
    	C = (double *)malloc( m*n*sizeof( double ) );

    	if (A == NULL || B == NULL || C == NULL) {
        	printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        	free(A);
        	free(B);
        	free(C);
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

		struct timeval start,finish;
		gettimeofday(&start,NULL);
    	for (r = 0; r < LOOP_COUNT; r++) {
        	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            	        m, n, p, alpha, A, p, B, n, beta, C, n);
    	}
		gettimeofday(&finish,NULL);
    	s_elapsed = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000 / LOOP_COUNT;

        double gflops = (2.0*s)*s*s/s_elapsed*1e-9;
		
    	printf ("%d\t%.5f\t%.5f\n", s, (s_elapsed * 1000),gflops);//unit: millisecond
    
    	free(A);
    	free(B);
    	free(C);
	}

    printf (" Test completed. \n\n");
    return 0;
}
