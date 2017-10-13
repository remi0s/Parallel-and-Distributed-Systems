#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include <omp.h>


#define MAXBINS 8


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

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;
  extern int sumthreads;
  //thread declerations
  extern int NUM_THREADS;
  extern int ACTIVE_THREADS;
  int NEW_THREAD = 0;
  int i=0;
  //if there's space for new threads, set flag to 1 and add the new threads to the count
  //once calling is over, decrement count

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

    //size of array for each thread
    int sizes[MAXBINS];
    int offsets[MAXBINS];
    for (i=0;i<MAXBINS;i++){ //its same like the original code, only into an array
      if(i==0){
        offsets[i]=0;
      }else{
        offsets[i]=BinSizes[i-1];
      }

      sizes[i] = BinSizes[i] - offsets[i];
    }




    #pragma omp flush(ACTIVE_THREADS) //synchronization of variable NEW_THREAD;
    //take care of thread max number to not be exceeded
    if (ACTIVE_THREADS < (NUM_THREADS) && 0 == NEW_THREAD){ //creation of new thread
        NEW_THREAD = 1;
    }
    if (ACTIVE_THREADS > (NUM_THREADS) && 1 == NEW_THREAD){ //deny creation of new thread
        NEW_THREAD = 0;
    }


      #pragma omp flush(NEW_THREAD) //synchronization of variable NEW_THREAD;
      /* Call the function recursively to split the lower levels */
      if(NEW_THREAD==1){ //If we havent reached the maximum allowd threads we create more threads


              #pragma omp parallel num_threads(2)
              {
              #pragma omp atomic //this declares that each thread need to run next command without interfierence of another thread, it like a mutex lock
              ACTIVE_THREADS++;

              #pragma omp flush(ACTIVE_THREADS) //synchronization of variable ACTIVE_THREADS
              int size=MAXBINS/2;


              #pragma omp for private(i) schedule(dynamic,size) //dynamic means that the loop will breake into
              for(i=0; i<MAXBINS; i++){                          // equal pieces and each thread will operate its part and then get assigned to another if there is left


                  //call recursively the radix sort;
                  truncated_radix_sort(&morton_codes[offsets[i]],
                        &sorted_morton_codes[offsets[i]],
                        &permutation_vector[offsets[i]],
                        &index[offsets[i]], &level_record[offsets[i]],
                        sizes[i],
                        population_threshold,
                        sft-3, lv+1);



                      }
            //each thread that return from the function called it , decrease the number of active threads so we can create more
              #pragma omp atomic //this delcares that each thread need to run next command alone, its like a lock
              ACTIVE_THREADS--;  //decreasing number of threads one by one till none left
              #pragma omp flush(ACTIVE_THREADS) //synchronization of ACTIVE_THREADS variable

            }

     }else{ //if we cant create more threads run this like in serial
       for(i=0; i<MAXBINS; i++){

         truncated_radix_sort(&morton_codes[offsets[i]],
                &sorted_morton_codes[offsets[i]],
                &permutation_vector[offsets[i]],
                &index[offsets[i]], &level_record[offsets[i]],
                sizes[i],
                population_threshold,
                sft-3, lv+1);
     }
    }
  }

}
