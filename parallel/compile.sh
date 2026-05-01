#!/bin/bash
gcc -O2 -fopenmp -shared -fPIC -o mlp_openmp.so mlp_openmp.c -lm
echo "Compiled: mlp_openmp.so"
