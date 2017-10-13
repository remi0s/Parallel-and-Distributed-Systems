#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

/* print the life board */

void print (int *board, int rows,int columns) {
  int   i, j;

  /* for each row */
  for (i=0; i<rows; i++) {
    /* print each column position... */
    for (j=0; j<columns ; j++) {
      printf ("%c", Board(i,j) ? 'x' : ' ');
      //printf ("%d", Board(i,j) ? Board(i,j) : 0); //to see the board as 0 and 1 , in order to check for bugs
    }

    /* followed by a carriage return */
    printf ("\n");
  }
}



/* display the table with delay and clear console */

void display_table(int *board, int rows,int columns) {
  print (board, rows,columns);
  //usleep(100000);
  //puts ("\033[H\033[J");
  /* clear the screen using VT100 escape codes */

}
