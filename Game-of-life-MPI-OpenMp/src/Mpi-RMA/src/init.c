#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <game-of-life.h>

/* set everthing to zero */

void initialize_board (int *board, int rows,int columns) {
  int   i, j;
  for (i=0; i<rows; i++)
    for (j=0; j<columns; j++)
      Board(i,j) = 0;
}

/* generate random table */

void generate_table (int *board, int rows,int columns, float threshold) {

  int rank,numtasks;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int  j,i;

  srand(time(NULL)+rank);
  // int k=rand();
  // printf("%d\n",k);

  for (i=0; i<rows; i++) {
    for (j=0; j<columns; j++) {
      Board(i,j) = ( (float)rand() / (float)RAND_MAX ) < threshold;
    }
  }
}
