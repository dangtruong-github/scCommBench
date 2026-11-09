#!/bin/bash

source /opt/intel/oneapi/setvars.sh
make && {
    mpirun -np 1 ./main --n-iter 1000 > test.log
}