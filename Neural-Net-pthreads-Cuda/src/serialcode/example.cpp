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
#include "sys/time.h"
using namespace std;
#include "string.h";

static float **pattern;
static float **desiredout;
// #define PATTERN_COUNT 4
// #define PATTERN_SIZE 2




int main(int argc, char *argv[])
{
  int NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,EPOCHS,NUM_THREADS,PATTERN_SIZE,PATTERN_COUNT,select,EXPORT_FILE;
  NETWORK_OUTPUT=1;
  select=1;
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
    select=1;
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
    select = 1;//atoi(argv[6]);     // Display output
    EXPORT_FILE = atoi(argv[7]);
    for(int i=0;i<HIDDEN_LAYERS;i++){
       hiddenlayerNeuronCount[i]=number_of_neurons;
       cout<<"HIDDEN_LAYERS["<<i<<"] has "<<hiddenlayerNeuronCount[i]<<" neurons"<<endl;
      }

  }else{
    printf("Usage: %s PATTERN_SIZE HIDDEN_LAYERS [NEURONS] EPOCHS NUM_THREADS EXPORT_FILE\n\n", argv[0]);
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

  printf("\n\t\tNET CONFIGURATIONS \n\n"
   "  PATTERN_SIZE                       : Number of inputs= %d\n"
   "  HIDDEN_LAYERS                      : numer of hiddenlayers= %d\n"
   "  [NEURONS](optional)                : number of hidenlayer neurons= %d\n"
   "  EPOCHS                             : number of EPOCHS= %d\n"
   "  NUM_THREADS                        : number of THREADS= %d\n"
   "  SELECT (2 for CPU ,3 for CUDA)     : CPU OR CUDA = %d\n"
   "  EXPORT_FILE                        : export?= %d\n\n"
   ,PATTERN_SIZE,HIDDEN_LAYERS,hiddenlayerNeuronCount[0],EPOCHS,NUM_THREADS,select,EXPORT_FILE);

  NETWORK_INPUTNEURONS=PATTERN_SIZE;
  PATTERN_COUNT=1<<PATTERN_SIZE;
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

    //bpnet net;
    bpnet net;

    //We create the network


    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);


    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);

    //Start the neural network training
    struct timeval startwtime, endwtime;
    double duration;
    gettimeofday (&startwtime, NULL);
    cout << "Start training for " << EPOCHS << " " << endl;
    int batch_size=100;
    int randomPattern;
    for(int i=0;i<EPOCHS;i++)
    {
        for(int j=0;j<PATTERN_COUNT;j++)
        {

            //cout<<"random= "<<randomPattern<<endl;
            net.batchTrain(desiredout[j],pattern[j]);

        }
        // randomPattern = rand()%(PATTERN_COUNT-0) + 0;
        // net.batchTrain(desiredout[randomPattern],pattern[randomPattern]);

          net.applyBatchCumulations(0.2f,0.1f);
          // if(i%10000==0){
          //   cout<<"\r"<<"Current EPOCH = "<<i<<flush;
          // }
    }
    gettimeofday (&endwtime, NULL);

    duration = (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    cout << "Train net duration : " << duration<< endl;
    //once trained test all patterns
    float square_error=0;
    for(int i=0;i<PATTERN_COUNT;i++)
    {

        net.propagate(pattern[i]);

    //display result
    square_error+=(*desiredout[i]-net.getOutput().neurons[0]->output)*(*desiredout[i]-net.getOutput().neurons[0]->output);
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< net.getOutput().neurons[0]->output << endl;
    }
    square_error/=PATTERN_COUNT;
    cout<<"square_error= "<<square_error<<endl;

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
