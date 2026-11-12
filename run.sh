#!/bin/bash

source /opt/intel/oneapi/setvars.sh
make && {
    mpirun -np 6 ./main --n-iter 1000 > out.log 2> err.log
}