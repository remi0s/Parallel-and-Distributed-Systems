# A simple bash script that for running experiments
# Note: To run the script make sure it you have execution rights
# (use: chmod u+x run_tests.sh to give execution rights)
!/bin/bash

NAME="7918"
DATE=$(date "+%Y-%m-%d-%H:%M:%S")
FILE_PREF=$NAME-cilk

echo $NAME
echo $DATE

make clean; make
# run cube experiments
for N in 1048576 2097152 4194304 8388608 16777216 33554432 ; do \
    for P in 128 ; do \
        for L in 18 ; do \
		for T in 2 4 8 16 32 64 128 256 512 1024 2048; do \
          		  echo sphere N=$N Threads=$T && ./test_octree $N 1 $P 2 $L $T >> $FILE_PREF-sphere.txt ; \
		done ; \
	done ; \
    done ; \
done ;
# run octant experiments
for N in 1048576 2097152 4194304 8388608 16777216 33554432 ; do \
    for P in 128 ; do \
        for L in 18 ; do \
		for T in 2 4 8 16 32 64 128 256 512 1024 2048; do \
	           echo cube N=$N Threads=$T && ./test_octree $N 0 $P 2 $L $T >> $FILE_PREF-cube.txt ; \
		done ; \
        done ; \
    done ; \
done ;
