#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
//#include <cilk/reducer_opadd.h> //needs to be included to use the addition reducer
#include <pthread.h>

#define MAXBINS 8
//CILK_C_REDUCER_OPADD(active_threads_radix, int, 0);
pthread_mutex_t mymutex=PTHREAD_MUTEX_INITIALIZER;

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

void truncated_radix_sort(unsigned long int *morton_codes,
			  unsigned long int *sorted_morton_codes,
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N,
			  int population_threshold,
			  int sft, int lv){

  extern int NUM_THREADS;
  extern int active_threads;
  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;
  int i=0,j=0;



  if(N<=0){

    return;
  }
  else if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv; // record the level of the node
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes

    return;
  }
  else{

    level_record[0] = lv;
    // Find which child each point belongs to
    for(j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)
    int offset = 0;
    for(i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }

    for(j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }

    //swap the index pointers
    swap(&index, &permutation_vector);

    //swap the code pointers
    swap_long(&morton_codes, &sorted_morton_codes);

    /* Call the function recursively to split the lower levels */
    int sizes[MAXBINS];
    int offsets[MAXBINS];
    for (i=0;i<MAXBINS;i++){
      if(i==0){
        offsets[i]=0;
      }else{
        offsets[i]=BinSizes[i-1];
      }
      sizes[i]=BinSizes[i]-offsets[i];
    }

    for (i=0; i<MAXBINS; i++)
    {
      // int offset = (i>0) ? BinSizes[i-1] : 0;
      // int size = BinSizes[i] - offset;

      if(active_threads<NUM_THREADS-1){ //check if we can create new threads
        //active_threads_radix.value++;

        pthread_mutex_lock(&mymutex); //we need mutex so we wont mess up active_threads count
        active_threads++;
        pthread_mutex_unlock(&mymutex);

        cilk_spawn truncated_radix_sort(&morton_codes[offsets[i]],
  			   &sorted_morton_codes[offsets[i]],
  			   &permutation_vector[offsets[i]],
  			   &index[offsets[i]], &level_record[offsets[i]],
  			   sizes[i],
  			   population_threshold,
  			   sft-3, lv+1);

      }else{
        //printf("in serial , active_threads_radix= %d\n",active_threads_radix);
        truncated_radix_sort(&morton_codes[offsets[i]],
  			   &sorted_morton_codes[offsets[i]],
  			   &permutation_vector[offsets[i]],
  			   &index[offsets[i]], &level_record[offsets[i]],
  			   sizes[i],
  			   population_threshold,
  			   sft-3, lv+1);
      }


    }
    //printf("reducer value =%d \n",active_threads_radix.value);
    cilk_sync;
    //printf("cilk spawn , active_threads_radix= %d\n",active_threads);



  }
}
