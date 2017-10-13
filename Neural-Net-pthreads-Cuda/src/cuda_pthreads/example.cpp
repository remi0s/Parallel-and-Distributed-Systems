/**********************************************************************************************************
  DEMO CODE: XOR Backpropagation Example
  File: example.cpp
  Version: 0.1
  Copyright(C) NeuroAI (http://www.learnartificialneuralnetworks.com)
  Documentation: http://www.learnartificialneuralnetworks.com/neural-network-software/backpropagation-source-code/
  Written by Daniel Rios <daniel.rios@learnartificialneuralnetworks.com> , June 2013

 /*********************************************************************************************************/

#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <ctime>
#include "bpnet.h"
#include <pthread.h>
#include "sys/time.h"

using namespace std;
#include "threadpool/ThreadPool.h"

// ThreadPool and Mutex
pthread_mutex_t mutexBusy;
static int mutexsum;
static int *mutextable;

// Net matrix as global so we can access it with out data_wraper
static bpnet *netMatrix;
static float **pattern;
static float **desiredout;
static int main_start_batch;
static int threads_start_batch;



struct data_wraper{
  int pattern_size,counter,nnet_id;

  data_wraper(){}
  ~data_wraper(){}

};

// not needed
const int MAX_TASKS = 4;

void train_wrapper(void* arg)
{

  data_wraper* data1 = (data_wraper*) arg;
  int i=data1->nnet_id*data1->counter;

  while(data1->counter >0){
    netMatrix[data1->nnet_id].batchTrain(desiredout[i],pattern[i]);
    i++;
    data1->counter--;
  }

  pthread_mutex_lock (&mutexBusy);

  mutexsum+=-1;

  if(data1->nnet_id!=0){
    mutextable[(data1->nnet_id)-1]=1;
  }

  pthread_mutex_unlock (&mutexBusy);

}

int main(int argc, char *argv[])
{
  int NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,EPOCHS,NUM_THREADS,PATTERN_SIZE,PATTERN_COUNT,select,EXPORT_FILE;
  NETWORK_OUTPUT=1;
  select=2;
  int *hiddenlayerNeuronCount;



  if (argc < 6) { // Check if the command line arguments are correct
    printf("Usage: %s PATTERN_SIZE HIDDEN_LAYERS [NEURONS] EPOCHS NUM_THREADS SELECT EXPORT_FILE\n\n", argv[0]);

    PATTERN_SIZE=3;
    HIDDEN_LAYERS=2;
    hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    hiddenlayerNeuronCount[0]=2;
    hiddenlayerNeuronCount[1]=2;
    EPOCHS=100000;
    NUM_THREADS=4;
    select=2;
    EXPORT_FILE=0;

    printf("Using default net configurations \n\n"
	   "  PATTERN_SIZE                       : Number of inputs= %d\n"
	   "  HIDDEN_LAYERS                      : numer of hiddenlayers= %d\n"
     "  [NEURONS](optional)                : number of hidenlayer neurons= %d\n"
     "  EPOCHS                             : number of EPOCHS= %d\n"
     "  NUM_THREADS                        : number of THREADS= %d\n"
     "  SELECT (2 for CPU ,3 for CUDA)     : CPU OR CUDA = %d\n"
     "  EXPORT_FILE                        : number of EPOCHS= %d\n\n"
     ,PATTERN_SIZE,HIDDEN_LAYERS,hiddenlayerNeuronCount[0],EPOCHS,select,NUM_THREADS,EXPORT_FILE);





  }else if (argc==7){
    PATTERN_SIZE = atoi(argv[1]);
    HIDDEN_LAYERS = atoi(argv[2]);
    EPOCHS = atoi(argv[3]);
    NUM_THREADS = atoi(argv[4]);
    select = atoi(argv[5]);
    EXPORT_FILE = atoi(argv[6]);

    if(HIDDEN_LAYERS>0){
      hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
      for(int i=0;i<HIDDEN_LAYERS;i++){
          printf("Enter number of neurons for %d hidden layer\n",(i+1));
          int number_of_neurons;
          cin >> number_of_neurons;
          hiddenlayerNeuronCount[i]=number_of_neurons;
          printf("HIDDEN_LAYERS[%d] has %d neurons \n",i,hiddenlayerNeuronCount[i]);
      }
    }
  }else if (argc==8){
    int number_of_neurons;

    PATTERN_SIZE = atoi(argv[1]);        // Rows
    HIDDEN_LAYERS = atoi(argv[2]);       //Columns
    hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    number_of_neurons = atoi(argv[3]);
    EPOCHS = atoi(argv[4]); // Propability of life cell
    NUM_THREADS = atoi(argv[5]);     // Display output
    select = atoi(argv[6]);     // Display output
    EXPORT_FILE = atoi(argv[7]);
    for(int i=0;i<HIDDEN_LAYERS;i++){
       hiddenlayerNeuronCount[i]=number_of_neurons;
       cout<<"HIDDEN_LAYERS["<<i<<"] has "<<hiddenlayerNeuronCount[i]<<" neurons"<<endl;
      }



  }else{
    printf("Usage: %s PATTERN_SIZE HIDDEN_LAYERS [NEURONS] EPOCHS NUM_THREADS SELECT EXPORT_FILE\n\n", argv[0]);
    printf(
	   "  PATTERN_SIZE                       : Number of inputs\n"
	   "  HIDDEN_LAYERS                      : numer of hiddenlayers\n"
     "  [NEURONS](optional)                : number of hidenlayer neurons\n"
     "  EPOCHS                             : number of EPOCHS\n"
     "  NUM_THREADS                        : number of THREADS\n"
     "  SELECT (2 for CPU ,3 for CUDA)     : CPU OR CUDA \n"
     "  EXPORT_FILE                        : number of EPOCHS\n\n");

     printf("Exiting programm \n");
     return(1);

  }

  NETWORK_INPUTNEURONS=PATTERN_SIZE;
  PATTERN_COUNT=1<<PATTERN_SIZE;

  printf("\n\t\tNET CONFIGURATIONS \n\n"
   "  PATTERN_SIZE                       : Number of inputs= %d\n"
   "  HIDDEN_LAYERS                      : numer of hiddenlayers= %d\n"
   "  [NEURONS](optional)                : number of hidenlayer neurons= %d\n"
   "  EPOCHS                             : number of EPOCHS= %d\n"
   "  NUM_THREADS                        : number of THREADS= %d\n"
   "  SELECT (2 for CPU ,3 for CUDA)     : CPU OR CUDA = %d\n"
   "  EXPORT_FILE                        : export = %d\n\n"
   ,PATTERN_SIZE,HIDDEN_LAYERS,hiddenlayerNeuronCount[0],EPOCHS,NUM_THREADS,select,EXPORT_FILE);
  cout<<"Number of patterns produced with "<<PATTERN_SIZE<<" is "<<PATTERN_COUNT<<endl;





    pattern=(float **)malloc(PATTERN_COUNT*(sizeof(float *)));
    for(int i=0;i<PATTERN_COUNT;i++){
      pattern[i]=(float *)malloc(PATTERN_SIZE*(sizeof(float)));
    }


    for(int i=0;i<(1<<PATTERN_SIZE);i++){
      for(int j=0;j<PATTERN_SIZE;j++){
           pattern[i][j]=i/(1<<(j))%2;
           //cout <<" "<<pattern[i][j]<<" ";
      }
    }


    desiredout=(float **)malloc(PATTERN_COUNT*(sizeof(float *)));
      for(int i=0;i<(1<<PATTERN_SIZE);i++){
        desiredout[i]=(float *)malloc(NETWORK_OUTPUT*(sizeof(float)));
      }


      for(int i=0;i<(1<<PATTERN_SIZE);i++){
        desiredout[i][0]= pattern[i][0];
        for(int j=1;j<PATTERN_SIZE;j++){
            desiredout[i][0]= (int)desiredout[i][0] ^(int)pattern[i][j];
        }
      }

  //cout << "Ccreating nnet = number of workers:"<<NUM_THREADS <<"Batch size="<<batch_size<< endl;
  //TODELET/bpnet *netMatrix=new bpnet[NUM_THREADS];//Our neural network object
  netMatrix=new bpnet[NUM_THREADS];
  cout << "Cloning the net maybe Completed l253"<< endl;
  netMatrix[0].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
  for(int i=1;i<NUM_THREADS;i++){
  netMatrix[i].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
  netMatrix[i].clone_bpnet(&netMatrix[0]);
  }

  //CHECKING FOR CUDA ERRORS
  int max_neurons=netMatrix[0].max_neuroncount();
  int max_inputs=netMatrix[0].max_inputcount();
  int number_of_layers=2+HIDDEN_LAYERS;
  int exitflag=1;
  for(int i=32;i>1;i--){
    if(((max_neurons*number_of_layers)%i)==0){
      exitflag=0;
      break;
    }
  }
  if(exitflag==0){
    for(int i=32;i>1;i--){
      if(((max_neurons*max_neurons*number_of_layers)%i)==0){
        exitflag=2;
        break;
      }
    }
  }
  if(exitflag==0){
    exitflag=1;
  }

  if(exitflag==1){
    printf("ERROR:\nCan't find proper grid for Cuda\nmodulo[max_neurons*number_of_layers,32:-1:2] must be 0 at least once\nExiting programm\n");
    exit(1);
  }





  int batch_per_thread=PATTERN_COUNT/NUM_THREADS;
  cout<<"Each thread must train "<<batch_per_thread<<" patterns"<<endl;
  if(batch_per_thread<0 || batch_per_thread%2!=0){
    cout<<endl<<endl;
    cout<<"ERROR:"<<endl;
    cout<<"For PATTERN_SIZE="<<PATTERN_SIZE<<" NUM_THREADS must be lower than "<<PATTERN_COUNT<<endl;
    cout<<"NUM_THREADS must be power of 2"<<endl;
    cout<<"EXIT PROGRAM"<<endl;
    exit(1);
  }

    //Init Thread Pool
    ThreadPool tp(NUM_THREADS-1);
    if(NUM_THREADS-1>0){
    int ret = tp.initialize_threadpool();
    if (ret == -1) {
      cerr << "Failed to initialize thread pool!" << endl;
      return 0;
      }
    }
    // init mutex to join
    pthread_mutex_init(&mutexBusy, NULL);
    mutexsum=0;
    mutextable =(int *)malloc(sizeof(int)*((NUM_THREADS)-1));

    //Start the neural network training

    cout << "Start training for " << EPOCHS << " " << endl;
    data_wraper data2;
    data2.nnet_id=0;
    data2.counter=batch_per_thread;
    data2.pattern_size=PATTERN_COUNT;
    main_start_batch=0;
    threads_start_batch=0;
    //data2=(data_wraper *)malloc(sizeof(data_wraper)*NUM_THREADS);

    // for(int i=0;i<NUM_THREADS;i++){
    //   data2[i].start=(int *)malloc(sizeof(int));
    // }


    int flag=1;
    struct timeval startwtime, endwtime;
    double duration;


    gettimeofday (&startwtime, NULL);



    for(int i=0;i<EPOCHS;i++)
    {
        pthread_mutex_lock (&mutexBusy);
        // reinitiallizing for synch method
        mutexsum=NUM_THREADS;
        //reinitiallizing for variable to cover such cost with gatherErrors2
        for(int j=0;j<NUM_THREADS-1;j++){
          mutextable[j]=0;
        }
        pthread_mutex_unlock (&mutexBusy);

        // Loop to send out work to thread pool
        for(int j=1;j<NUM_THREADS;j++)
        {
          // data2[j].nnet_id=j;
          // data2[j].counter=batch_per_thread;
          // data2[j].pattern_size=PATTERN_COUNT;
          //TODELETE/cout<<"Inside epoch="<<i<<"thread="<<j<<endl;
          data_wraper *data;
          data=new data_wraper();
          data->nnet_id=j;
          data->counter=batch_per_thread;
          data->pattern_size=PATTERN_COUNT;
          // data2[j].pattern_size=PATTERN_COUNT;
          //train_wrapper((void *) data);
          if(j!=0){
            //Task* t = new Task(&train_wrapper, (void*) &data2[j]);
            Task* t = new Task(&train_wrapper, (void*) data);
            tp.add_task(t);
          }
          //TODELETE/cout<<"Inside2 epoch="<<i<<"thread="<<j<<endl;
        }
        //TODELETE/cout<<"Outside epoch="<<i<<endl;
        data2.counter=batch_per_thread;
        train_wrapper((void *) &data2);
        //Synch method
        flag=1;
        while(flag)
        {
          if(pthread_mutex_trylock(&mutexBusy)==0)
          {
            if(mutexsum==0){
              flag=0;
              }
            pthread_mutex_unlock (&mutexBusy);
            for(int j=0;j<NUM_THREADS-1;j++){
              if(mutextable[j]==1){
                netMatrix[0].gatherErrors2(netMatrix,j+1);
                mutextable[j]=2;
              }
            }

            }
          }
      // // transfering new start to threads
      threads_start_batch=main_start_batch;
      // Finnishing gatherErrors2
      for(int j=0;j<NUM_THREADS-1;j++){
          if(mutextable[j]==1){
            //TODELETE/cout<<"Never inside"<<endl;
            netMatrix[0].gatherErrors2(netMatrix,j+1);
            mutextable[j]=2;
          }
        }

        if(select==2){
          netMatrix[0].applyBatchCumulations(0.2f,0.1f);
        }else if(select==3){
          netMatrix[0].KernelapplyBatchCumulations(0.2f,0.1f,max_neurons,max_inputs);
        }else{
          cout<<"EXIT: \n"<<"Last Argument must be either 2 for CPU applyBatch or 3 for CUDA applyBatch "<<endl;
          exit(-1);
        }


        for(int j=1;j<NUM_THREADS;j++){
        netMatrix[j].clone_bpnet(&netMatrix[0]);
        }
        //TODELETE/cout<<"Finishing epoch="<<i<<endl;
        // if(i%10000==0){
        //   cout<<"\r"<<"Current EPOCH = "<<i<<flush;
        // }
    }


    gettimeofday (&endwtime, NULL);

    duration = (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    cout << endl<< "Train net duration : " << duration<< endl;
    //once trained test all patterns
    float square_error=0;
    for(int i=0;i<PATTERN_COUNT;i++)
    {
        netMatrix[0].propagate(pattern[i]);
    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< netMatrix[0].getOutput().neurons[0]->output << endl;
        square_error+=(*desiredout[i]-netMatrix[0].getOutput().neurons[0]->output)*(*desiredout[i]-netMatrix[0].getOutput().neurons[0]->output);
    }
    square_error/=PATTERN_COUNT;
    cout<<"square_error= "<<square_error<<endl<<endl;
    tp.destroy_threadpool();

    //net complexity
    int count=PATTERN_SIZE+1;
    for(int i=0;i<HIDDEN_LAYERS;i++){
        count+=hiddenlayerNeuronCount[i];
    }
    /*		Creating FILE */
    //  result stable pattern

    FILE *result;
    char str[200];
    char snum[100];
    //Sequential choise =1
  if(EXPORT_FILE==0){
  }else if(EXPORT_FILE==1){
    sprintf(snum, "%d", select);
    strcpy(str,"resultStablePattern_");
    strcat(str, snum);
    strcat(str,"_");
    sprintf(snum, "%d", NUM_THREADS );
    strcat(str, snum);
    strcat(str,".txt");
    result=fopen(str,"a");
    long pos1=ftell(result);
    fseek(result, 0, SEEK_END);
    long pos2=ftell(result);
    if(pos1==0 && pos1==pos2){
      fprintf(result,"Duration  Inputsize NumNeuros NumThreads=%d\n",NUM_THREADS);
    }
    fseek(result, 0, SEEK_END);
    fprintf(result,"%f  %d %d\n",duration,PATTERN_SIZE,count);
  }else if(EXPORT_FILE==2){
    sprintf(snum, "%d", select);
    strcpy(str,"resultStableComplexity_");
    strcat(str, snum);
    strcat(str,"_");
    sprintf(snum, "%d", NUM_THREADS );
    strcat(str, snum);
    strcat(str,".txt");
    result=fopen(str,"a");
    long pos1=ftell(result);
    fseek(result, 0, SEEK_END);
    long pos2=ftell(result);
    if(pos1==0 && pos1==pos2){
      fprintf(result,"Duration  Inputsize NumNeuros NumThreads=%d\n",NUM_THREADS);
    }
    fseek(result, 0, SEEK_END);
    fprintf(result,"%f  %d %d\n",duration,PATTERN_SIZE,count);
    }

    if(EXPORT_FILE!=0){
        fclose(result);
    }


    return 0;
}
