#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <game-of-life.h>
#include <mpi.h>

void create_new_board(int i,int *board,int *newboard,int rows,int columns){
  int j, a;
  for (j=0; j<columns; j++) {
    a = adjacent_to (board, i, j, rows,columns);
    if (a == 2) NewBoard(i,j) = Board(i,j);
    if (a == 3) NewBoard(i,j) = 1;
    if (a < 2) NewBoard(i,j) = 0;
    if (a > 3) NewBoard(i,j) = 0;
  }
}


void play (int *board, int *newboard, int rows,int columns) {
  extern int NUM_THREADS;
  int   i,numtasks,lastrow=0,firstrow=0;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  if(numtasks>1){
    firstrow=1;
    lastrow=1;
  }
  /* for each cell, apply the rules of Life */
#pragma omp parallel for  private(i) shared(board,newboard)  num_threads(NUM_THREADS) //schedule(dynamic,size)
  for (i=firstrow; i<rows-lastrow; i++){
      create_new_board(i,board,newboard,rows,columns);
  }

}

void play_temp(int *board,int *temp_board,int *newboard,int rows,int columns,int offset){
  int i=1;
  int j,a;
  #pragma omp parallel for  private(j,a) shared(board,newboard) num_threads(NUM_THREADS)
  for (j=0; j<columns; j++) {
    a = adjacent_to (temp_board, i, j, rows,columns);
    if (a == 2) NewBoard(offset,j) = TempBoard(1,j);
    if (a == 3) NewBoard(offset,j) = 1;
    if (a < 2) NewBoard(offset,j) = 0;
    if (a > 3) NewBoard(offset,j) = 0;
  }

}

/*
  (copied this from some web page, hence the English spellings...)

  1.STASIS : If, for a given cell, the number of on neighbours is
  exactly two, the cell maintains its status quo into the next
  generation. If the cell is on, it stays on, if it is off, it stays off.

  2.GROWTH : If the number of on neighbours is exactly three, the cell
  will be on in the next generation. This is regardless of the cell's
  current state.

  3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will
  be off in the next generation.
*/
