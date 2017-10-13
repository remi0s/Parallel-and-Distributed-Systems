#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "utils.h"
#include <omp.h>

#define DIM 3

int main(int argc, char** argv){

  // Time counting variables
  struct timeval startwtime, endwtime;
  double hash_average=0,morton_average=0,sort_average=0,rearrange_average=0,total_average=0;
  extern int NUM_THREADS;
  extern int ACTIVE_THREADS;
  //omp_set_nested(1);  //allow nested threads

  sumthreads=0;

  if (argc != 7) { // Check if the command line arguments are correct
    printf("\nUsage: %s N dist pop rep L\n"
	   "where\n"
	   "N    : number of points\n"
	   "dist : distribution code (0-cube, 1-sphere)\n"
	   "pop  : population threshold\n"
	   "rep  : repetitions\n"
	   "L    : maximum tree height.\n", argv[0]);
    return (1);
  }

  // Input command line arguments
  int N = atoi(argv[1]); // Number of points
  int dist = atoi(argv[2]); // Distribution identifier
  int population_threshold = atoi(argv[3]); // populatiton threshold
  int repeat = atoi(argv[4]); // number of independent runs
  int maxlev = atoi(argv[5]); // maximum tree height
  NUM_THREADS=atoi(argv[6]);//Number of threads
  if(NUM_THREADS<=0){
    NUM_THREADS=omp_get_num_procs(); //defaut value NUM_THREADS
  }

  printf("Running for:\n Distribution code: %d \n Population threshold: %d \n Maximum height: %d \n Number of Particles: %d\n THREADS: %d\n",dist,population_threshold, maxlev,N,NUM_THREADS);

  float *X = (float *) malloc(N*DIM*sizeof(float));
  float *Y = (float *) malloc(N*DIM*sizeof(float));

  unsigned int *hash_codes = (unsigned int *) malloc(DIM*N*sizeof(unsigned int));
  unsigned long int *morton_codes = (unsigned long int *) malloc(N*sizeof(unsigned long int));
  unsigned long int *sorted_morton_codes = (unsigned long int *) malloc(N*sizeof(unsigned long int));
  unsigned int *permutation_vector = (unsigned int *) malloc(N*sizeof(unsigned int));
  unsigned int *index = (unsigned int *) malloc(N*sizeof(unsigned int));
  unsigned int *level_record = (unsigned int *) calloc(N,sizeof(unsigned int)); // record of the leaf of the tree and their level

  // initialize the index
  for(int i=0; i<N; i++){
    index[i] = i;
  }

  /* Generate a 3-dimensional data distribution */
  create_dataset(X, N, dist);

  /* Find the boundaries of the space */
  float max[DIM], min[DIM];
  find_max(max, X, N);
  find_min(min, X, N);

  int nbins = (1 << maxlev); // maximum number of boxes at the leaf level

  // Independent runs
  for(int it = 0; it<repeat; it++){
    ACTIVE_THREADS=0;

    gettimeofday (&startwtime, NULL);

    compute_hash_codes(hash_codes, X, N, nbins, min, max); // compute the hash codes

    gettimeofday (&endwtime, NULL);

    double hash_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    hash_average+=hash_time;
    printf("Time to compute the hash codes            : %fs\n", hash_time);





    gettimeofday (&startwtime, NULL);

    morton_encoding(morton_codes, hash_codes, N, maxlev); // computes the Morton codes of the particles

    gettimeofday (&endwtime, NULL);



    double morton_encoding_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    morton_average+=morton_encoding_time;

    printf("Time to compute the morton encoding       : %fs\n", morton_encoding_time);





    gettimeofday (&startwtime, NULL);

    // Truncated msd radix sort
    truncated_radix_sort(morton_codes, sorted_morton_codes,
			 permutation_vector,
			 index, level_record, N,
			 population_threshold, 3*(maxlev-1), 0);

    gettimeofday (&endwtime, NULL);

    double sort_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    sort_average+=sort_time;

    printf("Time for the truncated radix sort         : %fs\n", sort_time);



    gettimeofday (&startwtime, NULL);

    // Data rearrangement
    data_rearrangement(Y, X, permutation_vector, N);

    gettimeofday (&endwtime, NULL);


    double rearrange_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    rearrange_average+=rearrange_time;

    printf("Time to rearrange the particles in memory : %fs\n", rearrange_time);




    /* The following code is for verification */
    // Check if every point is assigned to one leaf of the tree


    int pass = check_index(permutation_vector, N);


    if(pass==N){
      printf("Index test PASS with %d of %d particles\n",pass,N);
    }
    else{
      printf("Index test FAIL with %d of %d particles\n",pass,N);
    }

    // Check is all particles that are in the same box have the same encoding.
    pass = check_codes(Y, sorted_morton_codes,
		       level_record, N, maxlev);

    if(pass==N){
      printf("Encoding test PASS with %d of %d particles\n\n\n",pass,N);
    }
    else{
      printf("Encoding test FAIL with %d of %d particles\n\n\n",pass,N);
    }

  }

  // Compute the average times for each function

  hash_average=hash_average/repeat;
  morton_average=morton_average/repeat;
  sort_average=sort_average/repeat;
  rearrange_average=rearrange_average/repeat;
  total_average=hash_average+morton_average+sort_average+rearrange_average;

  printf("Hash average time: %f \n",hash_average);
  printf("Morton average time: %f \n",morton_average);
  printf("Sort average time: %f \n",sort_average);
  printf("Rearrange average time: %f \n",rearrange_average);
  printf("Total_average: %f \n\n\n",total_average);

  /* clear memory */
  free(X);
  free(Y);
  free(hash_codes);
  free(morton_codes);
  free(sorted_morton_codes);
  free(permutation_vector);
  free(index);
  free(level_record);
}
