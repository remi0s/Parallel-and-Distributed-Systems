#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

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

void *cilk_quantize(void *arg){
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
  extern int NUM_THREADS;
  thread_args args[NUM_THREADS]; //create an array of thread_args for each thread arguments
  int i=0,j=0;
  int offset[NUM_THREADS]; //declaration of an array to keep the starting point of the proccess
  int size[NUM_THREADS]; //declaration of an array to keep track of how many particles each thread will quantize
  int sum=0;

  //assign a chunk of codes and mcodes to each thread
  offset[0] = 0;
  for (i = 0; i < NUM_THREADS -1; i++){
      size[i] = N/NUM_THREADS; //seperate the N particles into teams for each thread, for example 10 for each thread
      sum=sum + size[i];   	 // we keep track how many particles we seperated till now
      offset[i+1]=sum;       //store the sum into an array so we can use it to for each thread starting point
    }
    size[NUM_THREADS-1] = N - sum;  //last thread's number of particles


    //create NUM_THREADS threads
    for(i=0;i<NUM_THREADS;i++){
    //Give values to arg array valuables
    args[i].X=&X[DIM*offset[i]]; //
    args[i].codes=&codes[DIM*offset[i]]; //
    args[i].N=size[i];  //
    args[i].low=low;
    args[i].step=step;
    //thread creation(which thread,attribute,which function, arguments)
    cilk_spawn cilk_quantize((void *)&args[i]);
  }
  cilk_sync;

}

float max_range(float *x){
  int i=0;
  float max = -FLT_MAX;
  for(i=0; i<DIM; i++){
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
  int i=0;
  for(i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries
  }

  qstep = max_range(range) / nbins; // The quantization step

  quantize(codes, X, min, qstep, N); // Function that does the quantization

}
