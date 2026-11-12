#!/bin/bash

source /opt/intel/oneapi/setvars.sh
make && {
    mpirun -np 6 ./main --n-iter 100 -o data/test_100_2.bin > out.log 2> err.log
}