#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void helper(float const *J,float *If,float *Z,int size,int icenter,int jcenter,int xcenter,int ycenter,int rows,int columns,float const filtSigma,float const *H);
__device__ void adjFunction(float const *J,float *If,float *Z,int icenter ,int jcenter,float const filtSigma,int rows,int columns,int size,float const *H);




__global__ void nonLocalMeans(float const *J,float *If,float *Z,float const filtSigma,int const patchSize,int rows,int columns,float const *H) {




  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;


  if( p<rows && q<columns) {

      adjFunction(J,If,Z,p,q,filtSigma,rows,columns,patchSize,H);


  }

  __syncthreads();
}





__device__ void helper(float const *J,float *If,float *Z,int size,int icenter,int jcenter,int xcenter,int ycenter,int rows,int columns,float const filtSigma,float const *H){
  int i,j,z=0;
  float w=0;
  int ki,lj,kx,ly;
   for (i=-size/2; i<=size/2; i++){
     for (j=-size/2; j<=size/2; j++){
       //AN den yparxoun ta shmeia pairnoume ta diametrika tous
       if((icenter+i)>=0&&(icenter+i)<rows){
         ki=i;
       }else{
         ki=-i;
       }
       if((jcenter+j)>=0&&(jcenter+j)<rows){
         lj=j;
       }else{
         lj=-j;
       }

       if((xcenter+i)>=0&&(xcenter+i)<rows){
         kx=i;
       }else{
         kx=-i;
       }
       if((ycenter+j)>=0&&(ycenter+j)<rows){
         ly=j;
       }else{
         ly=-j;
       }
       w+=((J[(icenter+ki)*rows+(jcenter+lj)]-J[(xcenter+kx)*rows+(ycenter+ly)])*(J[(icenter+ki)*rows+(jcenter+lj)]-J[(xcenter+kx)*rows+(ycenter+ly)]))*(H[z]*H[z]);

       z++;
       }


   }
   w=exp(-(w*w)/(filtSigma));
   Z[icenter*rows+jcenter]+=w;
   If[icenter*rows+jcenter]+=w*J[xcenter*rows+ycenter];

}


__device__ void adjFunction(float const *J,float *If,float *Z,int icenter ,int jcenter,float const filtSigma,int rows,int columns,int size,float const *H){
  int i,j;
    for(i=0;i<rows;i++){
      for(j=0;j<columns;j++){
          helper(J,If,Z,size,icenter,jcenter,i,j,rows,columns,filtSigma,H);
      }
    }
    If[icenter*rows+jcenter]=If[icenter*rows+jcenter]/Z[icenter*rows+jcenter];


}
