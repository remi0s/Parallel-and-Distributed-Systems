#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <omp.h>


#define DIM 3


void data_rearrangement(float *Y, float *X,
			unsigned int *permutation_vector,
			int N){
	int i=0;
	extern int NUM_THREADS;
	int size=N/NUM_THREADS;
	
	#pragma omp parallel for private(i) num_threads(NUM_THREADS) schedule(dynamic,size)
  for(i=0; i<N; i++){
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }

}
