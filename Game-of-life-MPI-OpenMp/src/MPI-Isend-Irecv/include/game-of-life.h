/* #ifndef UTILS_H_   /\* Include guard *\/ */
/* #define UTILS_H_ */


#define NewBoard(x,y) newboard[(x)*columns + (y)]
#define TempBoard(x,y) temp_board[(x)*columns + (y)]
#define Board(x,y) board[(x)*columns + (y)]
/* set everthing to zero */

void initialize_board (int *board, int rows,int columns);

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N);

/* add to a height index, wrapping around */

int yadd (int i, int a, int N);

/* return the number of on cells adjacent to the i,j cell */

int adjacent_to (int *board, int i, int j, int rows,int columns);

/* play the game through one generation */

void play (int *board, int *newboard, int rows,int columns);
/*temporary play for the arrays from the other tasks */
void play_temp(int *board,int *temp_board,int *newboard,int rows,int columns,int offset);

/* print the life board */

void print (int *board, int rows,int columns);

/* generate random table */

void generate_table (int *board, int rows,int columns, float threshold);

/* display the table with delay and clear console */

void display_table(int *board, int rows,int columns);
void one_task(int *board,int *newboard,int rows,int columns,int disp,int t);
void more_than_one_tasks(int *board,int *newboard,int rows,int columns,int disp,int t);

int NUM_THREADS;
/* #endif // FOO_H_ */
