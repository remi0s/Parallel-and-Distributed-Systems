SHELL := /bin/bash # Use bash syntax

CC = sm_20

NVCC	= nvcc -arch=$(CC)
SAMPLE	= nonLocalMeansKernel
ARGS	= -ptx

all: sample

sample:
	$(NVCC) $(ARGS) $(SAMPLE).cu

clean:
	rm -rf *~ *.ptx
