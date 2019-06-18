#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 5:00
#SBATCH --job-name=Hmmmm...
#SBATCH -o /home-mscluster/zbowditch/slurm.%N.&j.out
#SBATCH -e /home-mscluster/zbowditch/slurm.%N.&j.err
srun ./MPI_NQP
