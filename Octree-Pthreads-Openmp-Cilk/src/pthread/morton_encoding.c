#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"

#define DIM 3


//define an struct for thread arguments
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
    x = (x | x << 2) & 0x1249249249249249; // 0000 0000 0001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

void *thread_morton(void *arg){
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
      pthread_exit(0);


}

/* The function that transform the morton codes into hash codes */
void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level){
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

      //create NUM_THREADS threads
      for(i=0;i<NUM_THREADS;i++){
      //Give values to arg array valuables
      args[i].mcodes=&mcodes[indicator[i]]; //we store into arguments mcodes the address of the starting point for each thread
      args[i].codes=&codes[DIM*indicator[i]]; //we store into arguments codes the address of the starting point for each thread, its dim*indicator because each code have x,y,z
      args[i].N=particles_number[i];  //we readress the number of particles for its thread
      args[i].max_level=max_level;
      //thread creation(which thread,attribute,which function, arguments)
      rc=pthread_create(&threads[i],&myattr,thread_morton,(void *)&args[i]);
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
          printf("ERROR; return code from pthread_join() morton is %d\n", rc);
          return;
          }

        }

}
