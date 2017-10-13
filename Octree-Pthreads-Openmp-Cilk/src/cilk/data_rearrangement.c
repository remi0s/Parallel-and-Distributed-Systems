#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


#define DIM 3
typedef struct {
	float *Y;
	float *X;
	unsigned int *permutation_vector;
	int N;

}thread_arguments;

void *cilk_rearrangement(void *args){
	thread_arguments *arguments=(thread_arguments *)args;
	float *Y=arguments->Y;
	float *X=arguments->X;
	unsigned int *permutation_vector=arguments->permutation_vector;
	int N=arguments->N;

	for(int i=0; i<N; i++){
		memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
	}

}

void data_rearrangement(float *Y, float *X, unsigned int *permutation_vector, int N){
				//threads and  declarations
				extern int NUM_THREADS;
				int i=0;


				thread_arguments args[NUM_THREADS]; //create an array of thread_args for each thread arguments
				int offset[NUM_THREADS];
				int size[NUM_THREADS];
				int sum=0;

				//assign a chunk of codes and mcodes to each thread
				offset[0] = 0;
				for (i = 0; i < NUM_THREADS -1; i++){
						size[i] = N/NUM_THREADS; //seperate the N particles into teams for each thread, for example 10 for each thread
						sum=sum + size[i];   	 // we keep track how many particles we seperated till now
						offset[i+1]=sum;       //store the sum into an array so we can use it to for each thread starting point
					}
			  size[NUM_THREADS-1] = N - sum;  //last thread's number of particles



				for(i=0;i<NUM_THREADS;i++){
					args[i].Y=&Y[offset[i]*DIM];
					args[i].X=X;
					args[i].permutation_vector=&permutation_vector[offset[i]];
					args[i].N=size[i];

	      cilk_spawn cilk_rearrangement((void *)&args[i]);
	     }
			 cilk_sync;

}
