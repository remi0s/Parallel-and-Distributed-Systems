#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "pthread.h"

#define DIM 3

// a struct for thread arguments
typedef struct{
  unsigned int *codes;
  float *X;
  float *low;
  float step;
  int N;
}thread_args;

inline unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);

}

void *thread_quantize(void *arg){
  //create a struct thread_args and  new variables that take struct variable's values
  thread_args *arguments=(thread_args *) arg;
  unsigned int *codes=arguments->codes;
  float *X=arguments->X;
  float *low=arguments->low;
  int N=arguments->N;
  float step=arguments->step;
  //original code from quantize
  for(int i=0; i<N; i++){
    for(int j=0; j<DIM; j++){
      codes[i*DIM + j] = compute_code(X[i*DIM + j], low[j], step);
    }
  }
}
/* Function that does the quantization */
void quantize(unsigned int *codes, float *X, float *low, float step, int N){
  //threads and  declarations
  extern int NUM_THREADS;
  int rc,i=0;  //rc:thread return, i: counter
  void *status; //a pointer for pthread_join function

  // Initialize and set thread joinable attribute
  pthread_t threads[NUM_THREADS];
  pthread_attr_t myattr;
  pthread_attr_init(&myattr);
  pthread_attr_setdetachstate(&myattr, PTHREAD_CREATE_JOINABLE);


  thread_args args[NUM_THREADS]; //create an array of thread_args for each thread arguments
  int indicator[NUM_THREADS]; //declaration of an array to keep the starting point of the proccess
  int particles_number[NUM_THREADS]; //declaration of an array to keep track of how many particles each thread will quantize
  int sum=0;

  //assign a chunk of codes and mcodes to each thread
  indicator[0] = 0;
  for (i = 0; i < NUM_THREADS -1; i++){
      particles_number[i] = N/NUM_THREADS; //seperate the N particles into teams for each thread, for example 10 for each thread
      sum=sum + particles_number[i];   	 // we keep track how many particles we seperated till now
      indicator[i+1]=sum;       //store the sum into an array so we can use it to for each thread starting point
    }
    particles_number[NUM_THREADS-1] = N - sum;  //last thread's number of particles

    //create NUM_THREADS threads
    for(i=0;i<NUM_THREADS;i++){
    //Give values to arg array valuables
    args[i].X=&X[DIM*indicator[i]]; //
    args[i].codes=&codes[DIM*indicator[i]]; //
    args[i].N=particles_number[i];  //
    args[i].low=low;
    args[i].step=step;
    //thread creation(which thread,attribute,which function, arguments)
    rc=pthread_create(&threads[i],&myattr,thread_quantize,(void *)&args[i]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      return;
      }
    }

    // release the attribute and w8 for the threads to join
    pthread_attr_destroy(&myattr);

    for(i=0; i<NUM_THREADS; i++) {
      rc = pthread_join(threads[i], &status);
      if (rc) {
          printf("ERROR; return code from pthread_join() is %d\n", rc);
          return;
          }

        }

}




float max_range(float *x){

  float max = -FLT_MAX;
  for(int i=0; i<DIM; i++){
    if(max<x[i]){
      max = x[i];
    }
  }

  return max;

}

void compute_hash_codes(unsigned int *codes, float *X, int N,
			int nbins, float *min,
			float *max){

  float range[DIM];
  float qstep;

  for(int i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries
  }

  qstep = max_range(range) / nbins; // The quantization step

  quantize(codes, X, min, qstep, N); // Function that does the quantization

}
