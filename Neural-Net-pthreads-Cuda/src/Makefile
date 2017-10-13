GCC=g++ -w -std=c++11
NVCC=nvcc -w -arch=sm_20 -Wno-deprecated-gpu-targets
PATH1=cuda_pthreads
PATH2=serialcode
OBJPATH=cuda_pthreads/bin/obj
ThreadPool_lib=cuda_pthreads/threadpool
Build_dir=cuda_pthreads/build
VERSION=0.04


all:
	$(NVCC) $(ThreadPool_lib)/CondVar.cpp -c -o $(Build_dir)/CondVar.o 
	$(NVCC) $(ThreadPool_lib)/Mutex.cpp -c -o $(Build_dir)/Mutex.o 
	$(NVCC) $(ThreadPool_lib)/Task.cpp -c -o $(Build_dir)/Task.o 
	$(NVCC) $(ThreadPool_lib)/ThreadPool.cpp -c -o $(Build_dir)/ThreadPool.o
	$(NVCC) $(PATH1)/bpnet.cpp -c -o $(Build_dir)/bpnet.o -x cu
	$(NVCC) $(PATH1)/example.cpp -c -o $(Build_dir)/example.o
	$(NVCC) $(Build_dir)/CondVar.o $(Build_dir)/Mutex.o $(Build_dir)/Task.o $(Build_dir)/ThreadPool.o $(Build_dir)/bpnet.o $(Build_dir)/example.o  -o backpropagation
	
	$(GCC) $(PATH2)/bpnet.cpp $(PATH2)/bpnet.h $(PATH2)/example.cpp -lm -o backpropagationserial

	#$(GCC) CondVar.o Mutex.o Task.o ThreadPool.o  $(ThreadPool_lib)/threadpool_test.cpp -lpthread -o threadpool_test

clean:
	rm -f *~ *.o *.bin backpropagation backpropagationserial
