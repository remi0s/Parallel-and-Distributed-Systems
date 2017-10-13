/**********************************************************************************************************
  C++ Source FILE: Backpropagation Neural Network Implementation
  File: bpnet.cpp
  Version: 0.1
  Copyright(C) NeuroAI (http://www.learnartificialneuralnetworks.com)
  Documentation:http://www.learnartificialneuralnetworks.com/neural-network-software/backpropagation-source-code/
  Written by Daniel Rios <daniel.rios@learnartificialneuralnetworks.com> , June 2013

 /*********************************************************************************************************/
#include "bpnet.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <pthread.h>
/*****************************neuron routines*******************************/

neuron::neuron():weights(0),deltavalues(0),output(0),gain(0),wgain(0) //constructor
{

}
//Destructor
neuron::~neuron()
{
    if(weights)
        delete [] weights;
    if(deltavalues)
        delete [] deltavalues;
}

//Initializates neuron weights
void neuron::create(int inputcount)
{
    assert(inputcount);
    float sign=-1;//to change sign
    float random;//to get random number
    weights=new float[inputcount];
    deltavalues=new float[inputcount];
    errorGain=0;
    errorWeight=0;

    //important initializate all weights as random unsigned values
    //and delta values as 0
    for(int i=0;i<inputcount;i++)
    {
        //get a random number between -0.5 and 0.5
        random=(float(rand()) / float(RAND_MAX))/1.f; //min 0.5
        random*=sign;
        sign*=-1;
        weights[i]=random;
        deltavalues[i]=0;
    }
    gain=-1;

    random=(float(rand()) / float(RAND_MAX))/1.f; //min 0.5
    random*=sign;
    sign*=-1;
    wgain=random;




}


// Copy method of neuron
void neuron::clone_neuron(int inputsize, struct neuron *main_neuron){

  this->output=main_neuron->output;
  this->gain=main_neuron->gain;
  this->wgain=main_neuron->wgain;
  //this->weights=new float[inputsize];
  this->errorGain=main_neuron->errorGain;
  this->errorWeight=main_neuron->errorWeight;
  //this->deltavalues=new float[inputsize];
  for(int i=0;i<inputsize;i++){
    this->weights[i]=main_neuron->weights[i];
    this->deltavalues[i]=main_neuron->deltavalues[i];
  }

}






/***********************************Layer member functions********************************/
layer::layer():neurons(0),neuroncount(0),layerinput(0),inputcount(0)
{

}

layer::~layer()
{
    if(neurons)
    {
        for(int i=0;i<neuroncount;i++)
        {
            delete neurons[i];
        }
        delete [] neurons;
    }
    if(layerinput)
    {
        delete [] layerinput;
    }
}

void layer::create(int inputsize, int _neuroncount)
{
    assert(inputsize && _neuroncount);//check for errors
    int i;
    neurons=new neuron*[_neuroncount];
    for(i=0;i<_neuroncount;i++)
    {
        neurons[i]=new neuron;
        neurons[i]->create(inputsize);
    }

    layerinput=new float[inputsize];
    neuroncount=_neuroncount;
    inputcount=inputsize;
}
//Calculates the neural network result of the layer using the sigmoid function
void layer::calculate()
{
    int i,j;
    float sum;
    //Apply the formula for each neuron
    for(i=0;i<neuroncount;i++)
    {
        sum=0;//store the sum of all values here
        for(j=0;j<inputcount;j++)
        {
        //Performing function
            sum+=neurons[i]->weights[j] * layerinput[j]; //apply input * weight
        }
        sum+=neurons[i]->wgain * neurons[i]->gain; //apply the gain or theta multiplied by the gain weight.
        //sigmoidal activation function
        neurons[i]->output= 1.f/(1.f + exp(-sum));//calculate the sigmoid function
       // neurons[i]->output=-1 + 2*(1.f + exp(-sum));
    }
}

// Clone the layer of a neural net
void layer::clone_layer(struct layer *main_layer){


  this->inputcount=main_layer->inputcount;
  this->neuroncount=main_layer->neuroncount;
  //this->layerinput=new float[this->inputcount];
  //this->neurons=new neuron*[this->neuroncount];
  for(int i=0;i<this->neuroncount;i++){
  //this->neurons[i]=new neuron;
  this->neurons[i]->clone_neuron(this->inputcount, main_layer->neurons[i]);
  }
}

// Get the error gain
void layer::getLayer(struct layer *main_layer){
  for(int i=0;i<this->neuroncount;i++){
    this->neurons[i]->errorGain+=main_layer->neurons[i]->errorGain;
    this->neurons[i]->errorWeight+=main_layer->neurons[i]->errorWeight;
  }
}


/***************************bpnet object functions**************/
bpnet::bpnet():m_hiddenlayers(0),m_hiddenlayercount(0)
{
}

bpnet::~bpnet()
{
    if(m_hiddenlayers)
    {
        for(int i=0;i<m_hiddenlayercount;i++)
        {
            delete m_hiddenlayers[i];
        }

        delete [] m_hiddenlayers;
    }

}

void bpnet::create(int inputcount, int inputneurons, int outputcount, int *hiddenlayers, int hiddenlayercount)
{
   //make sure required values are not zero
            assert(inputcount && inputneurons && outputcount);
            int i;
            m_inputlayer.create(inputcount,inputneurons);
            if(hiddenlayers && hiddenlayercount)
            {
                m_hiddenlayers=new layer*[hiddenlayercount];
                m_hiddenlayercount=hiddenlayercount;
                for(i=0;i<hiddenlayercount;i++)
                {
                    m_hiddenlayers[i]=new layer;
                    if(i==0)
                    {
                    //first hidden layer receives the output of the inputlayer so we set as input the neuroncount
                    //of the inputlayer
                        m_hiddenlayers[i]->create(inputneurons,hiddenlayers[i]);
                    }
                    else
                    {
                        m_hiddenlayers[i]->create(hiddenlayers[i-1],hiddenlayers[i]);
                    }
                }
                m_outputlayer.create(hiddenlayers[hiddenlayercount - 1],outputcount);
            }
            else
            {
                m_outputlayer.create(inputneurons,outputcount);
            }
            numberOfBatches=0;
}


void bpnet::propagate(const float *input)
{
    //The propagation function should start from the input layer
    //first copy the input vector to the input layer Always make sure the size
    //"array input" has the same size of inputcount
    memcpy(m_inputlayer.layerinput,input,m_inputlayer.inputcount * sizeof(float));
    //now calculate the inputlayer
    m_inputlayer.calculate();
    update(-1);//propagate the inputlayer out values to the next layer
    if(m_hiddenlayers)
    {
        //Calculating hidden layers if any
        for(int i=0;i<m_hiddenlayercount;i++)
        {
            m_hiddenlayers[i]->calculate();
            update(i);
        }
    }
    //calculating the final statge: the output layer
    m_outputlayer.calculate();
}
//Main training function. Run this function in a loop as many times needed per pattern
float bpnet::train(const float *desiredoutput, const float *input, float alpha, float momentum)
{
    //function train, teaches the network to recognize a pattern given a desired output
    float errorg=0; //general quadratic error
    float errorc; //local error;
    float sum=0,csum=0;
    float delta,udelta;
    float output;
    //first we begin by propagating the input
    propagate(input);
    int i,j,k;
    //the backpropagation algorithm starts from the output layer propagating the error  from the output
    //layer to the input layer


      for(i=0;i<m_outputlayer.neuroncount;i++)
      {
          //calculate the error value for the output layer
          output=m_outputlayer.neurons[i]->output; //copy this value to facilitate calculations
          //from the algorithm we can take the error value as
          errorc=(desiredoutput[i] - output) * output * (1 - output);
          //and the general error as the sum of delta values. Where delta is the squared difference
          //of the desired value with the output value
          //quadratic error
          errorg+=(desiredoutput[i] - output) * (desiredoutput[i] - output) ;

          //now we proceed to update the weights of the neuron
          for(j=0;j<m_outputlayer.inputcount;j++)
          {
              //get the current delta value
              delta=m_outputlayer.neurons[i]->deltavalues[j];
              //update the delta value
              udelta=alpha * errorc * m_outputlayer.layerinput[j] + delta * momentum;
              //update the weight values
              m_outputlayer.neurons[i]->weights[j]+=udelta;
              m_outputlayer.neurons[i]->deltavalues[j]=udelta;

              //we need this to propagate to the next layer
              sum+=m_outputlayer.neurons[i]->weights[j] * errorc;
          }

          //calculate the weight gain
          m_outputlayer.neurons[i]->wgain+= alpha * errorc * m_outputlayer.neurons[i]->gain;

      }
      for(i=(m_hiddenlayercount - 1);i>=0;i--)
      {
          for(j=0;j<m_hiddenlayers[i]->neuroncount;j++)
          {
              output=m_hiddenlayers[i]->neurons[j]->output;
              //calculate the error for this layer
              errorc= output * (1-output) * sum;
              //update neuron weights
              for(k=0;k<m_hiddenlayers[i]->inputcount;k++)
              {
                  delta=m_hiddenlayers[i]->neurons[j]->deltavalues[k];
                  udelta= alpha * errorc * m_hiddenlayers[i]->layerinput[k] + delta * momentum;
                  m_hiddenlayers[i]->neurons[j]->weights[k]+=udelta;
                  m_hiddenlayers[i]->neurons[j]->deltavalues[k]=udelta;
                  csum+=m_hiddenlayers[i]->neurons[j]->weights[k] * errorc;//needed for next layer

              }

              m_hiddenlayers[i]->neurons[j]->wgain+=alpha * errorc * m_hiddenlayers[i]->neurons[j]->gain;

          }
          sum=csum;
          csum=0;
      }
      //and finally process the input layer
      for(i=0;i<m_inputlayer.neuroncount;i++)
      {
          output=m_inputlayer.neurons[i]->output;
          errorc=output * (1 - output) * sum;

          for(j=0;j<m_inputlayer.inputcount;j++)
          {
              delta=m_inputlayer.neurons[i]->deltavalues[j];
              udelta=alpha * errorc * m_inputlayer.layerinput[j] + delta * momentum;
              //update weights
              m_inputlayer.neurons[i]->weights[j]+=udelta;
              m_inputlayer.neurons[i]->deltavalues[j]=udelta;
          }
          //and update the gain weight
          m_inputlayer.neurons[i]->wgain+=alpha * errorc * m_inputlayer.neurons[i]->gain;
      }




    //return the general error divided by 2
    return errorg / 2;


}



void bpnet::batchTrain(const float *desiredoutput, const float *input)
{
    //function train, teaches the network to recognize a pattern given a desired output
    float errorc; //local error;
    float sum=0,csum=0;
    //float delta,udelta;
    float output;
    //first we begin by propagating the input
    propagate(input);
    int i,j,k;
    //the backpropagation algorithm starts from the output layer propagating the error  from the output
    //layer to the input layer




      for(i=0;i<m_outputlayer.neuroncount;i++)
      {
          //calculate the error value for the output layer
          output=m_outputlayer.neurons[i]->output; //copy this value to facilitate calculations
          //from the algorithm we can take the error value as
          errorc=(desiredoutput[i] - output) * output * (1 - output);
          m_outputlayer.neurons[i]->errorGain+=errorc;

          //and the general error as the sum of delta values. Where delta is the squared difference
          //of the desired value with the output value
          //quadratic error


          //now we proceed to update the weights of the neuron
          for(j=0;j<m_outputlayer.inputcount;j++)
          {
              m_outputlayer.neurons[i]->errorWeight+=errorc*m_outputlayer.layerinput[j];
              sum+=m_outputlayer.neurons[i]->weights[j] * errorc;
          }



      }

      for(i=(m_hiddenlayercount - 1);i>=0;i--)
      {
          for(j=0;j<m_hiddenlayers[i]->neuroncount;j++)
          {
              output=m_hiddenlayers[i]->neurons[j]->output;
              //calculate the error for this layer
              errorc=output * (1-output) * sum;
              m_hiddenlayers[i]->neurons[j]->errorGain+= errorc;
              //update neuron weights
              for(k=0;k<m_hiddenlayers[i]->inputcount;k++)
              {
                  m_hiddenlayers[i]->neurons[j]->errorWeight+=errorc*m_hiddenlayers[i]->layerinput[k];

                  csum+=m_hiddenlayers[i]->neurons[j]->weights[k] * errorc;//needed for next layer

              }

          }
          sum=csum;
          csum=0;
      }
      //and finally process the input layer
      for(i=0;i<m_inputlayer.neuroncount;i++)
      {
          output=m_inputlayer.neurons[i]->output;
          errorc=output * (1 - output) * sum;
          m_inputlayer.neurons[i]->errorGain+=errorc;
          for (j=0;j<m_inputlayer.inputcount;j++){
            m_inputlayer.neurons[i]->errorWeight+=errorc*m_inputlayer.layerinput[j];
          }

      }


      numberOfBatches++; //keep track of how many batches passed



}
// A function to gather all errors from all nets
void bpnet::gatherErrors(class bpnet *netMatrix,int netCount)
{
  //calculate the error value for the output layer
  for(int i=1;i<netCount;i++){
      this->m_outputlayer.getLayer(&netMatrix[i].m_outputlayer);
      for(int j=0;j<m_hiddenlayercount;j++){
        this->m_hiddenlayers[j]->getLayer(netMatrix[i].m_hiddenlayers[j]);

      }
      this->m_inputlayer.getLayer(&netMatrix[i].m_inputlayer);
      this->numberOfBatches+=netMatrix[i].get_numberOfBatches(); //keep track of how many batches passed

  }


}
// A function to gather all errors from a specific nets
void bpnet::gatherErrors2(class bpnet *netMatrix,int specificNet){
  this->m_outputlayer.getLayer(&netMatrix[specificNet].m_outputlayer);
  for(int j=0;j<m_hiddenlayercount;j++){
    this->m_hiddenlayers[j]->getLayer(netMatrix[specificNet].m_hiddenlayers[j]);

  }
  this->m_inputlayer.getLayer(&netMatrix[specificNet].m_inputlayer);
  this->numberOfBatches+=netMatrix[specificNet].get_numberOfBatches(); //keep track of how many batches passed

}




__global__ void cuda_layer_comulations2(float alpha,float momentum,int numberOfBatches,float *weights,float *errorWeight,float *deltavalues)
{
  //int index=threadIdx.x*blockDim.y+threadIdx.y; //blockDim.y=max_neurons*max_inputs
  int index=blockDim.y*(blockIdx.y +gridDim.y*(threadIdx.x+blockIdx.x*blockDim.x))+threadIdx.y;



  float udelta=alpha *(errorWeight[index] /numberOfBatches) + deltavalues[index] * momentum;

  weights[index]+=udelta;
  deltavalues[index]=udelta;

  // if(index==0){
  //   weights[index]=blockDim.x;
  // }
  //
  // if(index==1){
  //   weights[index]=blockDim.y;
  // }


}




__global__ void cuda_layer_comulations(float alpha,int numberOfBatches,float *wgain,float *errorGain)
{
  //int index=threadIdx.x*blockDim.y+threadIdx.y; //blockDim.y=max_neurons, blockDim.x=number_of_layers
  int index=blockDim.y*(blockIdx.y +gridDim.y*(threadIdx.x+blockIdx.x*blockDim.x))+threadIdx.y;
  wgain[index]-=alpha*errorGain[index]/numberOfBatches;
  // if(index==0){
  //   wgain[index]=blockDim.y;
  // }


}

__global__ void cuda_TEST(float *wgain)
{
  //int index=threadIdx.x*blockDim.y+threadIdx.y; //blockDim.y=max_neurons, blockDim.x=number_of_layers
  int index=blockDim.y*(blockIdx.y +gridDim.y*(threadIdx.x+blockIdx.x*blockDim.x))+threadIdx.y;
  wgain[index]=index;
  // if(index==0){
  //   wgain[index]=blockDim.y;
  // }


}

int bpnet::max_neuroncount(){
  int max=m_outputlayer.neuroncount;
  for(int i=0;i<m_hiddenlayercount;i++){
    if(max<m_hiddenlayers[i]->neuroncount) max=m_hiddenlayers[i]->neuroncount;
  }
  if(max<m_inputlayer.neuroncount) max=m_inputlayer.neuroncount;
  return max;
}

int bpnet::max_inputcount(){
  int max=m_outputlayer.inputcount;
  for(int i=0;i<m_hiddenlayercount;i++){
    if(max<m_hiddenlayers[i]->inputcount) max=m_hiddenlayers[i]->inputcount;
  }
  if(max<m_inputlayer.inputcount) max=m_inputlayer.inputcount;
  return max;
}

void gridDimensions(int max_neurons,int number_of_layers,int *blocknum_x,int *blocknum_y,int *threadnum_x,int *threadnum_y){
  //blockDim.y
  if(max_neurons<32){
    *threadnum_y=max_neurons;
    *blocknum_y=1;
  }else{
    *blocknum_y=1;
    *threadnum_y=1;
    while((*blocknum_y)*(*threadnum_y)<max_neurons){
      *threadnum_y=32;
      while((*threadnum_y)>1){
        (*threadnum_y)--;
        if((*blocknum_y)*(*threadnum_y)==max_neurons){
          break;
        }
      }

      if((*blocknum_y)*(*threadnum_y)==max_neurons){
        break;
      }
      (*blocknum_y)++;

    }
  }


  //blockdim.x

  int max_xthread=floor(1024/(*threadnum_y));
  if(number_of_layers<max_xthread){
    (*blocknum_x)=1;
    (*threadnum_x)=number_of_layers;
  }else{
    (*blocknum_x)=1;
    (*threadnum_x)=max_xthread;

    while((*blocknum_x)*(*threadnum_x)<number_of_layers){
      (*threadnum_x)=max_xthread;
      while((*threadnum_x)>1){
        (*threadnum_x)--;
        if((*blocknum_x)*(*threadnum_x)==number_of_layers){
          break;
        }
      }

      if((*blocknum_x)*(*threadnum_x)==number_of_layers){
        break;
      }
      (*blocknum_x)++;

    }
  }




}


void bpnet::KernelapplyBatchCumulations(float alpha, float momentum,int max_neurons,int max_inputs)
{
    int i,j,k;

    //   clock_t start;
    //  double duration=0;

    int number_of_layers=m_hiddenlayercount+2;

    float *deltavalues_C;
    float *weights_C;
    float *wgain_C;
    float *errorWeight_C;
    float *errorGain_C;

    cudaMalloc((void **)&deltavalues_C,(number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float));
    cudaMalloc((void **)&weights_C,(number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float));
    cudaMalloc((void **)&errorWeight_C,(number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float));

    cudaMalloc((void **)&wgain_C,(number_of_layers)*(max_neurons)*sizeof(float));
    cudaMalloc((void **)&errorGain_C,(number_of_layers)*(max_neurons)*sizeof(float));

    float *deltavalues_D;
    deltavalues_D=(float *)malloc(number_of_layers*(max_neurons)*(max_inputs)*sizeof(float));

    float *weights_D;
    weights_D=(float *)malloc((number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float ));

    float *errorWeight_D;
    errorWeight_D=(float *)malloc(number_of_layers*(max_neurons)*(max_inputs)*sizeof(float ));

    float *wgain_D;
    wgain_D=(float *)malloc(number_of_layers*max_neurons*(sizeof(float )));

    float *errorGain_D;
    errorGain_D=(float *)malloc(number_of_layers*max_neurons*(sizeof(float )));



    layer *current_layer;
    int layercounter=1;
    for(k=0;k<number_of_layers;k++){

        if(k==0){
          current_layer=&m_outputlayer;
        }else if(k==number_of_layers-1){
          current_layer=&m_inputlayer;
        }else{
          current_layer=m_hiddenlayers[m_hiddenlayercount-layercounter];
          layercounter++;
        }


        for(i=0;i<max_neurons;i++)
        {

            if(i<current_layer->neuroncount){
              wgain_D[k*max_neurons+i]=current_layer->neurons[i]->wgain;
              errorGain_D[k*max_neurons+i]=current_layer->neurons[i]->errorGain;
            }else{
              wgain_D[k*max_neurons+i]=0;
              errorGain_D[k*max_neurons+i]=0;
            }

            //now we proceed to update the weights of the neuron
            for(j=0;j<max_inputs;j++)
            {
              if(j<current_layer->inputcount && i<current_layer->neuroncount ){
                errorWeight_D[(k*max_neurons+i)*max_inputs+j]=current_layer->neurons[i]->errorWeight;
                deltavalues_D[(k*max_neurons+i)*max_inputs+j]=current_layer->neurons[i]->deltavalues[j];
                weights_D[(k*max_neurons+i)*max_inputs+j]=current_layer->neurons[i]->weights[j];
              }else{
                errorWeight_D[(k*max_neurons+i)*max_inputs+j]=0;
                deltavalues_D[(k*max_neurons+i)*max_inputs+j]=0;
                weights_D[(k*max_neurons+i)*max_inputs+j]=0;
              }

            }


        }
    }

    int threadnum_x,threadnum_y,blocknum_x,blocknum_y;


    if(number_of_layers*max_neurons<1024){// (number_of_layers<32&&max_neurons<32)
      //den exei testaristei arketa
      // gridDimensions( 242, 50,&blocknum_x,&blocknum_y,&threadnum_x,&threadnum_y);
      // printf("threadnum_x=%d threadnum_y=%d blocknum_x=%d blocknum_y=%d\n",threadnum_x,threadnum_y,blocknum_x,blocknum_y);
      threadnum_x=number_of_layers;
      threadnum_y=max_neurons;
      blocknum_x=1;
      blocknum_y=1;
    }else{
      //den exei testaristei arketa
      gridDimensions( max_neurons, number_of_layers,&blocknum_x,&blocknum_y,&threadnum_x,&threadnum_y);

    }

    dim3 threadsPerBlock(threadnum_x,threadnum_y);
    dim3 numblocks(blocknum_x,blocknum_y);


    cudaMemcpy(wgain_C, wgain_D, (number_of_layers)*(max_neurons)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(errorGain_C, errorGain_D, (number_of_layers)*(max_neurons)*sizeof(float), cudaMemcpyHostToDevice);
    cuda_layer_comulations<<<numblocks,threadsPerBlock>>>(alpha, numberOfBatches, wgain_C,errorGain_C);



    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(wgain_D, wgain_C, (number_of_layers)*(max_neurons)*sizeof(float), cudaMemcpyDeviceToHost);

    if(number_of_layers*max_neurons*max_inputs<1024){//// //(number_of_layers<32 && max_neurons*max_inputs<32)
      //den exei testaristei arketa
      // gridDimensions( 242, 50,&blocknum_x,&blocknum_y,&threadnum_x,&threadnum_y);
      // printf("threadnum_x=%d threadnum_y=%d blocknum_x=%d blocknum_y=%d\n",threadnum_x,threadnum_y,blocknum_x,blocknum_y);
      threadnum_x=number_of_layers;
      threadnum_y=max_neurons*max_inputs;
      blocknum_x=1;
      blocknum_y=1;
    }else{
      //den exei testaristei arketa
      gridDimensions( max_neurons*max_inputs, number_of_layers,&blocknum_x,&blocknum_y,&threadnum_x,&threadnum_y);

    }


    dim3 threadsPerBlock2(threadnum_x,threadnum_y);
    dim3 numblocks2(blocknum_x,blocknum_y);

    cudaMemcpy(weights_C, weights_D, (number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(errorWeight_C, errorWeight_D, (number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deltavalues_C, deltavalues_D, (number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float), cudaMemcpyHostToDevice);
    cuda_layer_comulations2<<<numblocks2,threadsPerBlock2>>>( alpha, momentum, numberOfBatches, weights_C, errorWeight_C, deltavalues_C);

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(weights_D, weights_C, (number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(deltavalues_D, deltavalues_C, (number_of_layers)*(max_neurons)*(max_inputs)*sizeof(float), cudaMemcpyDeviceToHost);

    layercounter=1;
    for(k=0;k<number_of_layers;k++){



        if(k==0){
          current_layer=&m_outputlayer;
        }else if(k==number_of_layers-1){
          current_layer=&m_inputlayer;
        }else{

          current_layer=m_hiddenlayers[m_hiddenlayercount-layercounter];
          layercounter++;
        }


        for(i=0;i<max_neurons;i++)
        {

            if(i<current_layer->neuroncount){
              current_layer->neurons[i]->wgain=wgain_D[k*max_neurons+i];
              current_layer->neurons[i]->errorWeight=0;
              current_layer->neurons[i]->errorGain=0;
            }

            //now we proceed to update the weights of the neuron
            for(j=0;j<max_inputs;j++)
            {
              if(j<current_layer->inputcount && i<current_layer->neuroncount ){
                current_layer->neurons[i]->deltavalues[j]=deltavalues_D[(k*max_neurons+i)*max_inputs+j];
                current_layer->neurons[i]->weights[j]=weights_D[(k*max_neurons+i)*max_inputs+j];
              }

            }
        }
    }






    // //testing
    // int testsize_x,testsize_y;
    // testsize_x=271;
    // testsize_y=1;
    // if((testsize_x<32)&&(testsize_y<32)){
    //   //den exei testaristei arketa
    //   // gridDimensions( 242, 50,&blocknum_x,&blocknum_y,&threadnum_x,&threadnum_y);
    //   // printf("threadnum_x=%d threadnum_y=%d blocknum_x=%d blocknum_y=%d\n",threadnum_x,threadnum_y,blocknum_x,blocknum_y);
    //   threadnum_x=testsize_x;
    //   threadnum_y=testsize_y;
    //   blocknum_x=1;
    //   blocknum_y=1;
    // }else{
    //   //den exei testaristei arketa
    //   gridDimensions( testsize_y, testsize_x,&blocknum_x,&blocknum_y,&threadnum_x,&threadnum_y);
    //
    //
    // }
    // float *test_C;
    // cudaMalloc((void **)&test_C,(testsize_x)*(testsize_y)*sizeof(float));
    // float *test_D;
    // test_D=(float *)malloc(testsize_x*testsize_y*(sizeof(float )));
    //
    // for(k=0;k<testsize_x;k++){
    //   for(i=0;i<threadnum_y;i++){
    //     test_D[k*threadnum_y+i]=100.0;
    //   }
    // }
    //
    // dim3 threadsPerBlock3(threadnum_x,threadnum_y);
    // dim3 numblocks3(blocknum_x,blocknum_y);
    //
    //
    // cudaMemcpy(test_C, test_D, (testsize_x)*(testsize_y)*sizeof(float), cudaMemcpyHostToDevice);
    // cuda_TEST<<<numblocks3,threadsPerBlock3>>>(test_C);
    //
    //
    //
    // error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //   // print the CUDA error message and exit
    //   printf("CUDA error: %s\n", cudaGetErrorString(error));
    //   exit(-1);
    // }
    //
    // cudaDeviceSynchronize();
    // cudaMemcpy(test_D, test_C, (testsize_x)*(testsize_y)*sizeof(float), cudaMemcpyDeviceToHost);
    // // printf("\n\n");
    // // for(k=0;k<testsize_x;k++){
    // //   for(i=0;i<testsize_y;i++){
    // //     printf("test[%d]=%f\n",k*testsize_y+i,test_D[k*testsize_y+i] );
    // //   }
    // // }
    //
    //     cudaFree( test_C);




    cudaFree( deltavalues_C );
    cudaFree( weights_C );
    cudaFree( wgain_C );
    cudaFree( errorWeight_C );
    cudaFree( errorGain_C );


    numberOfBatches=0;




}






void bpnet::applyBatchCumulations(float alpha, float momentum)
{
    int i,j,k;
    float delta=0,udelta=0;


    for(i=0;i<m_outputlayer.neuroncount;i++)
    {



        //now we proceed to update the weights of the neuron
        for(j=0;j<m_outputlayer.inputcount;j++)
        {
            //get the current delta value
            delta=m_outputlayer.neurons[i]->deltavalues[j];
            //update the delta value
            udelta=alpha *(m_outputlayer.neurons[i]->errorWeight / (float)numberOfBatches) + delta * momentum;
            //update the weight values
            m_outputlayer.neurons[i]->weights[j]+=udelta;
            m_outputlayer.neurons[i]->deltavalues[j]=udelta;


        }

        //calculate the weight gain
        m_outputlayer.neurons[i]->wgain+= alpha * (m_outputlayer.neurons[i]->errorGain / (float)numberOfBatches) * m_outputlayer.neurons[i]->gain;
        m_outputlayer.neurons[i]->errorGain=0;
        m_outputlayer.neurons[i]->errorWeight=0;

    }

    for(i=(m_hiddenlayercount - 1);i>=0;i--)
    {
        for(j=0;j<m_hiddenlayers[i]->neuroncount;j++)
        {

            //update neuron weights
            for(k=0;k<m_hiddenlayers[i]->inputcount;k++)
            {
                delta=m_hiddenlayers[i]->neurons[j]->deltavalues[k];
                udelta= alpha * (m_hiddenlayers[i]->neurons[j]->errorWeight / (float)numberOfBatches)  + delta * momentum;
                m_hiddenlayers[i]->neurons[j]->weights[k]+=udelta;
                m_hiddenlayers[i]->neurons[j]->deltavalues[k]=udelta;


            }

            m_hiddenlayers[i]->neurons[j]->wgain+=alpha * (m_hiddenlayers[i]->neurons[j]->errorGain / (float)numberOfBatches) * m_hiddenlayers[i]->neurons[j]->gain;
            m_hiddenlayers[i]->neurons[j]->errorGain=0;
            m_hiddenlayers[i]->neurons[j]->errorWeight=0;
        }

    }

    //and finally process the input layer
    for(i=0;i<m_inputlayer.neuroncount;i++)
    {



        for(j=0;j<m_inputlayer.inputcount;j++)
        {
            delta=m_inputlayer.neurons[i]->deltavalues[j];
            udelta=alpha * (m_inputlayer.neurons[i]->errorWeight / (float)numberOfBatches)  + delta * momentum;
            //update weights
            m_inputlayer.neurons[i]->weights[j]+=udelta;
            m_inputlayer.neurons[i]->deltavalues[j]=udelta;
        }
        //and update the gain weight
        m_inputlayer.neurons[i]->wgain+=alpha * (m_inputlayer.neurons[i]->errorGain / (float)numberOfBatches) * m_inputlayer.neurons[i]->gain;
        m_inputlayer.neurons[i]->errorGain=0;
        m_inputlayer.neurons[i]->errorWeight=0;
    }



 numberOfBatches=0;

}

void bpnet::update(int layerindex)
{
    int i;

    if(layerindex==-1)
    {
        //dealing with the inputlayer here and propagating to the next layer

        for(i=0;i<m_inputlayer.neuroncount;i++)
        {

            if(m_hiddenlayers)//propagate to the first hidden layer
            {

                m_hiddenlayers[0]->layerinput[i]=m_inputlayer.neurons[i]->output;

            }
            else //propagate directly to the output layer
            {


                m_outputlayer.layerinput[i]=m_inputlayer.neurons[i]->output;

            }


        }

    }
    else
    {
        for(i=0;i<m_hiddenlayers[layerindex]->neuroncount;i++)
        {

            //not the last hidden layer
            if(layerindex < m_hiddenlayercount -1)
            {
                m_hiddenlayers[layerindex + 1]->layerinput[i]=m_hiddenlayers[layerindex]->neurons[i]->output;
            }
            else
            {
                m_outputlayer.layerinput[i]=m_hiddenlayers[layerindex]->neurons[i]->output;
            }
        }
    }

}


// method to get the number of hidden layer
int bpnet::get_m_hiddenlayercount(){
  return this->m_hiddenlayercount;
};
// method to get the number of batches
float bpnet::get_numberOfBatches(){
  return this->numberOfBatches;
};


// method to copy bpnet
void bpnet::clone_bpnet(class bpnet *main_bpnet)
{

  this->m_hiddenlayercount=main_bpnet->get_m_hiddenlayercount();
  this->m_inputlayer.clone_layer(&main_bpnet->m_inputlayer);
  this->numberOfBatches=main_bpnet->get_numberOfBatches();
  //this->m_hiddenlayers=new layer*[this->m_hiddenlayercount];

  for(int i=0;i<this->m_hiddenlayercount;i++){
    //this->m_hiddenlayers[i]=new layer;
        this->m_hiddenlayers[i]->clone_layer(main_bpnet->m_hiddenlayers[i]);
  }

  this->m_outputlayer.clone_layer(&main_bpnet->m_outputlayer);

}
