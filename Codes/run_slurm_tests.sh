#!/bin/bash

#SBATCH -p batch
#SBATCH -N 8
#SBATCH -t 1:00:00
#SBATCH --job-name=ha ha
#SBATCH -o /home-mscluster/zbowditch/slurm.%N.&j.out
#SBATCH -e /home-mscluster/zbowditch/slurm.%N.&j.err

srun -n 8 MPI_NQP
