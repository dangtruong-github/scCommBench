#!/bin/bash

source /opt/intel/oneapi/setvars.sh
make && {
    mpirun -np 2 ./main --n-iter 1000 > test.log
}