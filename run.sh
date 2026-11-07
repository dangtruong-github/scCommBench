#!/bin/bash

source /opt/intel/oneapi/setvars.sh
make && {
    mpirun -np 2 ./main --n-iter 10 > test.log
}