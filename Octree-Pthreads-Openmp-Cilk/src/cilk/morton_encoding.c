#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define DIM 3

typedef struct {
  unsigned long int *mcodes;
  unsigned int *codes;
  int N;
  int max_level;
}thread_args;

inline unsigned long int splitBy3(unsigned int a){
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}


void *cilk_morton(void *arg){
    //create a struct thread_args and  new variables that take struct variable's values
    thread_args *arguments=(thread_args*) arg;
    unsigned long int *mcodes= arguments->mcodes;
    unsigned int *codes= arguments->codes;
    int N=arguments->N;
    int max_level=arguments->max_level;

    for(int i=0; i<N; i++){
        // Compute the morton codes from the hash codes using the magicbits method
        mcodes[i] = mortonEncode_magicbits(codes[i*DIM], codes[i*DIM + 1], codes[i*DIM + 2]);
      }

}

/* The function that transform the morton codes into hash codes */
void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level){
  int i=0;
  extern int active_threads;
  extern int NUM_THREADS;

  thread_args args[NUM_THREADS];

  int offset[NUM_THREADS];
  int size[NUM_THREADS];
  offset[0]=0;
  int sum=0;
  for(i=0;i<NUM_THREADS-1;i++){
    size[i]=N/NUM_THREADS;
    sum+=size[i];
    offset[i+1]=sum;
  }
  size[NUM_THREADS-1]=N- sum;



  for( i=0; i<NUM_THREADS; i++){
    args[i].mcodes=&mcodes[offset[i]]; //we store into arguments mcodes the address of the starting point for each thread
    args[i].codes=&codes[DIM*offset[i]]; //we store into arguments codes the address of the starting point for each thread, its dim*indicator because each code have x,y,z
    args[i].N=size[i];  //we readress the number of particles for its thread
    args[i].max_level=max_level;
    // Compute the morton codes from the hash codes using the magicbits mathod
    cilk_spawn cilk_morton((void *)&args[i]);
    }

  cilk_sync;

}
