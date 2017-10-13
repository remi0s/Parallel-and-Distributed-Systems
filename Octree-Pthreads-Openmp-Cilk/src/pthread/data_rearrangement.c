#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pthread.h"


#define DIM 3

typedef struct {
	float *Y;
	float *X;
	unsigned int *permutation_vector;
	int N;

}thread_arguments;

void *parallel_rearrangement(void *args){
	thread_arguments *arguments=(thread_arguments *)args;
	float *Y=arguments->Y;
	float *X=arguments->X;
	unsigned int *permutation_vector=arguments->permutation_vector;
	int N=arguments->N;

	for(int i=0; i<N; i++){
		memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
	}
	pthread_exit(0);

}


void data_rearrangement(float *Y, float *X, unsigned int *permutation_vector, int N){
				//threads and  declarations
				extern int NUM_THREADS;
				int rc,i=0;  //rc:thread return, i: counter
				void *status; //a pointer for pthread_join function

				// Initialize and set thread joinable attribute
				pthread_t threads[NUM_THREADS];
				pthread_attr_t myattr;
				pthread_attr_init(&myattr);
				pthread_attr_setdetachstate(&myattr, PTHREAD_CREATE_JOINABLE);


				thread_arguments args[NUM_THREADS]; //create an array of thread_args for each thread arguments
				int indicator[NUM_THREADS];
				int particles_number[NUM_THREADS];
				int sum=0;

				//assign a chunk of codes and mcodes to each thread
				indicator[0] = 0;
				for (i = 0; i < NUM_THREADS -1; i++){
						particles_number[i] = N/NUM_THREADS; //seperate the N particles into teams for each thread, for example 10 for each thread
						sum=sum + particles_number[i];   	 // we keep track how many particles we seperated till now
						indicator[i+1]=sum;       //store the sum into an array so we can use it to for each thread starting point
					}
			  particles_number[NUM_THREADS-1] = N - sum;  //last thread's number of particles


				// for(int i=0; i<N; i++){
				// 	memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
				// }


				for(i=0;i<NUM_THREADS;i++){
					args[i].Y=&Y[indicator[i]*DIM];
					args[i].X=X;
					args[i].permutation_vector=&permutation_vector[indicator[i]];
					args[i].N=particles_number[i];

	      //thread creation(which thread,attribute,which function, arguments)
	      		rc=pthread_create(&threads[i],&myattr,parallel_rearrangement,(void *)&args[i]);
	      		if (rc) {
	        		printf("ERROR; return code from pthread_create() is %d\n", rc);
	        		return;
	        	}
	     }

	    // Free attribute and wait for the other threads
	    pthread_attr_destroy(&myattr);

	    for(i=0; i<NUM_THREADS; i++) {
	      rc = pthread_join(threads[i], &status);
	      if (rc) {
	          printf("ERROR; return code from pthread_join() is %d\n", rc);
	          return;
	          }

	        }




}
