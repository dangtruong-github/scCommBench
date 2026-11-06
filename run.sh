#!/bin/bash

make && {
    ./main --n-iter 1000 > test.log
}