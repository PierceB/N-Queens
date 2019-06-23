////////////////////////////////////////////////////////////
README (HPC Project 2019, Pierce Burke, Zachary Bowditch)
////////////////////////////////////////////////////////////

The make file and run.sh will compile and run the following files:
SERIAL_RE.c: This code executes the serial implementation using a recursive function to conduct the backtracking.
 
SERIAL_IT.c: Like the recursive method above, this code file will execute a backtracking approach to the N-Queens solution-finding problem using a single function instance and board array, conducting the backtracking iteratively as opposed to recursively.

CUDA_NQ.cu: This code will execute our CUDA implementation of the N-Queens solution-finding problem. This was run on an NVIDIA GPU with the specifications delineated in the report. 

MPI_NQP.c: Using the recursive method in SERIAL_RE, this code will assign to each of the specified number of threads (up to N / board length) a number of partially solved boards up to a depth of 1 (First column complete) using a round robin distribution scheme as described in the report.

Compile with: 

make

/////// Run examples of all scripts (Serial recursive, serial iterative, CUDA, MPI) with:

chmod +x run.sh
./run.sh

/////// Run slurm test file (on cluster) with:

sbatch run_slurm_tests.sh

/////// Clean the object and executable files with:

make clean
