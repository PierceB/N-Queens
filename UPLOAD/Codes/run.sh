#!/bin/bash

printf "Serial (Recursive)\n\n"
./SERIAL_RE.out 8 5 
echo ""
printf "Serial (Iterative)\n\n"
./SERIAL_IT.out
echo ""
printf  "CUDA (Depth 4)\n\n"
./CUDA_NQ.out 4
echo ""
printf  "MPI (4 Threads)\n\n"
mpiexec -n 4 ./MPI_NQP.out
echo ""
