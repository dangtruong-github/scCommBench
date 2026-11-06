#!/bin/bash

make && {
    ./main --n-iter 30 > test.log
}