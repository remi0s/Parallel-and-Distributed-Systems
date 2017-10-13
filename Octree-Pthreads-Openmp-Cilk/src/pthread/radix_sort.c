#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "pthread.h"

#define MAXBINS 8
extern int NUM_THREADS;
pthread_t *thread;
extern pthread_mutex_t mymutex;
pthread_attr_t myattr;


 //Define a struct so we count active_threads
 // typedef struct{
 //   int active_threads;
 // }thread_manager;
 //
 // thread_manager ids;
//Define a struct for thread arguments for radix sort parallel algorithm
 typedef struct {
   unsigned long int *morton_codes;
   unsigned long int *sorted_morton_codes;
   unsigned int *permutation_vector;
   unsigned int *index;
   unsigned int *level_record;
   int N;
   int population_threshold;
   int sft ;
   int lv;
 }thread_arguments;

inline void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

void *parallel_radix_sort(void *args){
  //create a new thread_arguments and transfer args values into this new struct
  thread_arguments *arguments=(thread_arguments *) args;
  //create new varieables and transfer the struct values into them
  unsigned long int *morton_codes=arguments->morton_codes;
  unsigned long int *sorted_morton_codes=arguments->sorted_morton_codes;
  unsigned int *permutation_vector=arguments->permutation_vector;
  unsigned int *index=arguments->index;
  unsigned int *level_record=arguments->level_record;
  int N=arguments->N;
  int population_threshold=arguments->population_threshold;
  int sft=arguments->sft ;
  int lv=arguments->lv;


  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;
  extern int NUM_THREADS;
  extern int active_threads;



  level_record[0] = lv;
  if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes
    return ;
  }
  else{

    thread_arguments new_args[MAXBINS];
    int rc,sum;
    void *status;
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }

    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }

    //swap the index pointers

    swap(&index, &permutation_vector);

    //swap the code pointers

    swap_long(&morton_codes, &sorted_morton_codes);


    //Calculate new_args[MAXBINS] so we can use it later
    for(int i=0; i<MAXBINS; i++){
      int offset = (i>0) ? BinSizes[i-1] : 0;
      int size = BinSizes[i] - offset;

      new_args[i].morton_codes=&morton_codes[offset];
      new_args[i].sorted_morton_codes=&sorted_morton_codes[offset];
      new_args[i].permutation_vector=&permutation_vector[offset];
      new_args[i].index=&index[offset];
      new_args[i].level_record=&level_record[offset];
      new_args[i].N=size;
      new_args[i].population_threshold=population_threshold;
      new_args[i].sft=sft-3;
      new_args[i].lv=lv+1;
    }

    for(int i=0;i<MAXBINS;i++){
      if(active_threads<NUM_THREADS){ //check if we can create more threads
        //if we can create more threads increase number of active threads by one and create a new thread
        pthread_mutex_lock(&mymutex); //we need mutex so we wont mess up active_threads count
        int thread_id=active_threads;
        active_threads++;
        pthread_mutex_unlock(&mymutex);

        rc=pthread_create(&thread[thread_id],&myattr,parallel_radix_sort,(void *)&new_args[i]);
        if (rc) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          return;
        }

        pthread_attr_destroy(&myattr);
        rc=pthread_join(thread[thread_id],NULL); //join the thread created with this thread_id
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            return;
        }
        }else {
          parallel_radix_sort((void *)&new_args[i]);
        }
      }
  }
}


void truncated_radix_sort(unsigned long int *morton_codes,
			  unsigned long int *sorted_morton_codes,
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N,
			  int population_threshold,
			  int sft, int lv){

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;
  extern int NUM_THREADS;
  extern int active_threads;

  if(NUM_THREADS<MAXBINS){  //this way we assure that we have empty thread slots for the starting threads
    active_threads=NUM_THREADS; //if we didnt do this, after the 1st time an starting thread created, it might create more threads causing
  }else{                        //to not have slot for the other starting threads
    active_threads=MAXBINS;
  }



  if(N<=0){

    return ;
  }
  else if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv;// record the level of the node
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes

    return ;
  }
  else{
    //threads and  declarations
    level_record[0] = lv;

    pthread_attr_init(&myattr); //intiallize attribes and mutexes
    pthread_mutex_init(&mymutex,NULL);
    pthread_attr_setdetachstate(&myattr,PTHREAD_CREATE_JOINABLE); //set threads joinable

    extern int NUM_THREADS;

    thread=(pthread_t *)malloc((NUM_THREADS)*sizeof(pthread_t));


    thread_arguments args[MAXBINS]; //create 8 arguments , one for each recursion

    int rc,i=0;  //rc:thread return, i: counter
    void *status; //a pointer for pthread_join function



    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }

    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }

    //swap the index pointers
    swap(&index, &permutation_vector);

    //swap the code pointers
    swap_long(&morton_codes, &sorted_morton_codes);


    //calculate arguments so we can use them later
    for(int i=0; i<MAXBINS; i++){
      int offset = (i>0) ? BinSizes[i-1] : 0;
      int size = BinSizes[i] - offset;

      args[i].morton_codes=&morton_codes[offset];
      args[i].sorted_morton_codes=&sorted_morton_codes[offset];
      args[i].permutation_vector=&permutation_vector[offset];
      args[i].index=&index[offset];
      args[i].level_record=&level_record[offset];
      args[i].N=size;
      args[i].population_threshold=population_threshold;
      args[i].sft=sft-3;
      args[i].lv=lv+1;
    }
    /* Call the parallel function to split the lower levels  */
    int starting_threads=0;
    for(int i=0;i<MAXBINS;i++){
      if(starting_threads<active_threads){ //we want to create max 8 starting threads
        starting_threads++;

        rc=pthread_create(&thread[i],&myattr,parallel_radix_sort,(void *)&args[i]);
        if (rc) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          return;
        }

      }else{ //if we can not create more threads , for example NUM_THREADS was 2, we run the parallel_radix_sort serially for the other 6 recursions
        parallel_radix_sort((void *)&args[i]);
      }

    }
    pthread_attr_destroy(&myattr);
    for(int i=0;i<starting_threads;i++){
      rc=pthread_join(thread[i],&status); //join the starting threads
      if (rc) {
          printf("ERROR; return code from pthread_join() is %d\n", rc);
          return ;
    }

    }
    pthread_mutex_destroy(&mymutex);
  }
  free(thread);
}
