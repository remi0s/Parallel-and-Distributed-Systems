/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <game-of-life.h>
#include <mpi.h>
#include <omp.h>


void swap(int **board, int **newboard){ //function to swap the adresses of board and newboard for the next round
  int *temp = *board;
  *board = *newboard;
  *newboard = temp;
}



int main (int argc, char *argv[]) {
  struct timeval start, finish,startgen,finishgen,startplay,finishplay; //variables to measure time
  extern int NUM_THREADS;
  int   *board, *newboard;
  int rank,numtasks;





  if (argc != 7) { // Check if the command line arguments are correct
    printf("Usage: %s Nx Ny thres t disp threads\n"
	   "where\n"
	   "  Nx     : number of rows\n"
     "  Ny     : number of columns\n"
	   "  thres : propability of alive cell\n"
     "  t     : number of generations\n"
	   "  disp  : {1: display output, 0: hide output}\n"
     "  threads  : number of threads for openMP\n"
     , argv[0]);
    return (1);
  }


  MPI_Init(&argc,&argv);  //MPI initialization
  gettimeofday (&start, NULL);      //Start time of the programm
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);   //numtasks take how many tasks where created
  if(numtasks<=0){
    printf("tasks number <=0 , ERROR");
    exit(0);
  }

  // Input command line arguments
  int Nx = atoi(argv[1]);        // Rows
  int Ny = atoi(argv[2]);       //Columns
  double thres = atof(argv[3]); // Propability of life cell
  int t = atoi(argv[4]);        // Number of generations
  int disp = atoi(argv[5]);     // Display output
  NUM_THREADS=atoi(argv[6]); //number of threads to use


  if(rank==0){
    printf("\nSize %dx%d with propability: %0.1f%%\n", Nx, Ny, thres*100);
    printf("Tasks number %d\n\n",numtasks);
  } //print info about the programm running

  int rows=Nx/numtasks; //Divide the rows into numtasks pieces
  int columns=Ny;      //so each task will create rows*columns board , and all together will be Nx*Ny
  board =NULL;        //Clean the pointer
  newboard= NULL;
  board = (int *)malloc(rows*columns*sizeof(int));      //memmory allocation for board and new board

  if (board == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }
  newboard=(int *)malloc(rows*columns*sizeof(int));
  if (newboard == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }


  gettimeofday (&startgen, NULL);

  generate_table(board,rows,columns,thres); //board generation

  gettimeofday (&finishgen, NULL);




  gettimeofday (&startplay, NULL);
  if(numtasks==1){             //if only one task was created by user then no need for all the sends and recieves.

    one_task(board,newboard,rows,columns,disp,t);

  }else{              //if more than one tasks where created

    more_than_one_tasks(board,newboard,rows,columns,disp,t);

  }
  gettimeofday (&finishplay, NULL);

  gettimeofday (&finish, NULL);           //total runtime counter


  double total_time = (double)((finish.tv_usec - start.tv_usec)
				/1.0e6 + finish.tv_sec - start.tv_sec);
  double generate_time = (double)((finishgen.tv_usec - startgen.tv_usec)
      	/1.0e6 + finishgen.tv_sec - startgen.tv_sec);
  double play_time = (double)((finishplay.tv_usec - startplay.tv_usec)
        /1.0e6 + finishplay.tv_sec - startplay.tv_sec);


  printf("Rank=%d Generate board time : %f\n",rank,generate_time);
  printf("Rank=%d Play time           : %f\n",rank,play_time);
  printf("Rank=%d Total run time      : %f\n\n",rank,total_time);

  free(board);
  free(newboard);
  MPI_Finalize();
}



/* Function for only one task */
void one_task(int *board,int *newboard,int rows,int columns,int disp,int t){
  int i;
  for (i=0; i<t; i++) {
    if (disp){
      display_table (board,rows,columns);
      usleep(100000);                    //function for a better view of the board when printed, give some delay
      puts ("\033[H\033[J");  //clean the board.
    }


    play (board, newboard, rows, columns);     //play function , the last variable 0 is ment to be as an starting point inside for.
                                                    //with 0 it plays from 0 to rows and 0 to columns , the whole board!
    swap(&board,&newboard);   //swap the boards so the board takes the new values
  }
}





/*Function for more than one tasks */
void more_than_one_tasks(int *board,int *newboard,int rows,int columns,int disp,int t){
  int j,k,rank,numtasks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);      //rank take the  taskid
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  int prev;  //an variable which shows who is my neighbour task above me
  int next; //an variable which shows who is my neighbour task bellow me
  if(rank==0){
    prev=numtasks-1;   //for 1st task the above neighbour is the last task, in order to achieve circular play in board
    next=rank+1;      //for 1st task below neighbour is the next task (rank=1)
  }else if(rank==numtasks-1){
    prev=rank-1;  //for last task the above neightbour is the previous rank
    next=0;       //and the bellow neightbour is the 1st rank because we want to play in circural mode.
  }else{          //for all the other ranks
    prev=rank-1;  //previous rank
    next=rank+1;   //next rank
  }
  //create needed pointers and initialize them
  int *send_top_array, *send_bottom_array, *recieve_top_array, *recieve_bottom_array, *temp_board;
  send_top_array=send_bottom_array=recieve_top_array=recieve_bottom_array=temp_board=NULL;
  //memmory allocation for the above pointers
  send_top_array=(int *)malloc(columns*sizeof(int));
  send_bottom_array=(int*)malloc(columns*sizeof(int));
  temp_board=(int *)malloc(3*columns*sizeof(int));
  recieve_top_array=(int *)malloc(columns*sizeof(int));
  recieve_bottom_array=(int *)malloc(columns*sizeof(int));
                                                       //3*columns because i use it in order to play a temporary board which contains the
                                                      //row that i'm interested in, and 2 more rows, the one above and the one bellow

  for(k=0;k<t;k++){              //t Generations
      #pragma omp parallel for  private(j) shared(board)  num_threads(NUM_THREADS)
      for(j=0;j<columns;j++){
        send_top_array[j]=Board(0,j); //send_top_array=1st row
        send_bottom_array[j]=Board(rows-1,j); //send bottom_array=last row
      }
      MPI_Request sendreq[2];  //create 2 requests and 2 stats to use in isend and irecv
      MPI_Request recvreq[2];
      MPI_Status waitsend[2];
      MPI_Status waitrecv[2];


      MPI_Isend (send_bottom_array,columns,MPI_INT,next,2,MPI_COMM_WORLD,&sendreq[0]);//send whenever u can bottom array to the next rank with code 2
      MPI_Irecv (recieve_top_array,columns,MPI_INT,prev,2,MPI_COMM_WORLD,&recvreq[0]);
      MPI_Isend (send_top_array,columns,MPI_INT,prev,1,MPI_COMM_WORLD,&sendreq[1]); // send whenever u can top array to the previous rank with code 1
      MPI_Irecv (recieve_bottom_array,columns,MPI_INT,next,1,MPI_COMM_WORLD,&recvreq[1]);

      play (board, newboard, rows,columns); //play all the board except 1st and last row (because we need to get info about the neighbour cells for this two rows)

      MPI_Wait(&sendreq[0],&waitsend[0]);
      MPI_Wait(&recvreq[0],&waitrecv[0]);

      #pragma omp parallel for  private(j) shared(board)  num_threads(NUM_THREADS)
      for(j=0;j<columns;j++){
        TempBoard(0,j)=recieve_top_array[j];//create an temp_board which has 1st row=neighbour row we recieved
        TempBoard(1,j)=Board(0,j);          //2ond row the row we want to change
        TempBoard(2,j)=Board(1,j);          //3rd row the row bellow the one we want to change
      }
      play_temp(board,temp_board,newboard,3,columns,0); //play to change the 1st row now that we recieve the info from the previous rank!
                                                           // 0 means that this play will change the Board(0,j),which is the 1st row

      MPI_Wait(&sendreq[1],&waitsend[1]);
      MPI_Wait(&recvreq[1],&waitrecv[1]);
      #pragma omp parallel for  private(j) shared(board)  num_threads(NUM_THREADS)
      for(j=0;j<columns;j++){
        TempBoard(0,j)=Board(rows-2,j);//create an temp_board which has 1st row the row above the one we want to change
        TempBoard(1,j)=Board(rows-1,j);//2ond row the row we want to change
        TempBoard(2,j)=recieve_bottom_array[j];//3rd row the row bellow the one we want to change , which is the one we recieved
      }
      play_temp(board,temp_board,newboard,3,columns,rows-1);//play to change the last row


      MPI_Barrier(MPI_COMM_WORLD);          //synchronize all the tasks so we can now display the table without problems
      swap(&board,&newboard); //swap the boards so the board takes the new value

      int flag=1;  //just a variable nonesense i use to send and recieve, in order to 1st rank displays 1st , 2ond rank displays 2ond ...etc
      MPI_Status dispstat;
      if(disp){   //if user selected to display
        if(rank==0){ //if this is the 1st rank then
          display_table(board,rows,columns); //display your table.
          MPI_Send (&flag,1,MPI_INT,next,1,MPI_COMM_WORLD); //send a message to the next rank , to show that u ended the display
          MPI_Recv (&flag,1,MPI_INT,prev,1,MPI_COMM_WORLD,&dispstat); //recieve a message from the last rank so you can clear the screen
          usleep(100000);                                             //just an delay for better view of the table
          if(k!=t-1){                                               //makes the board not to be cleaned on last generation to check for mistakes easier
            puts ("\033[H\033[J");                         //clean the board if its not the last turn
          }

        }else{                //if rank!=0 then
          MPI_Recv (&flag,1,MPI_INT,prev,1,MPI_COMM_WORLD,&dispstat);//wait until you recieve from previous rank a message
          display_table(board,rows,columns);    //then display the board
          MPI_Send (&flag,1,MPI_INTEGER,next,1,MPI_COMM_WORLD); //and send a message to the next one to start displaying
        }

      }
      MPI_Barrier(MPI_COMM_WORLD);  //synchronize again the tasks, this is needed because sometimes without it the display is messed up


  }

  free(send_top_array);
  free(send_bottom_array);
  free(recieve_bottom_array);
  free(temp_board);
  free(recieve_top_array);

}
