// CUDA approach for solving the N-Queens problem (Pierce Burke and Zachary Bowditch)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>

// include CUDA files
#include <cuda_runtime.h>
#include "inc/helper_functions.h"
#include "inc/helper_cuda.h"

#define N 15

int no_solutions; // increment every time a solution is found

int save_data(const char *filename, int *souls, double *times, int n){ //Helper function used to save the data as a file
	FILE *f;
	f = fopen(filename, "w");
	for(int i = 0; i < n; i++){
		fprintf(f, "%d:%f:%d\n", i, times[i], souls[i]);
	}
	fclose(f);
	return 1;
}

int is_valid(int* sol, int n, int row, int col){                   //function to check if a queen can be placed in a square without conflict
	if(row >= n) return 0;
	for(int c = 0; c <= col; c++){ //Use C++ in C :..)
		if(sol[c] == row) return 0;
		if(sol[c] - c == row - col) return 0;
		if(n - sol[c] - c == n - row - col) return 0;
	}
	return 1;
}

long factorial(long num){     //calculate factorial, used for upperbound of partial solutions
	if(num == 0) return 1;
	return num*factorial(num - 1);
}

__device__ int d_is_valid(int* sol, int n, int row, int col){  //Device function for kernel  to check if queen can be placed
	if(row >= n) return 0;
	for(int c = 0; c <= col; c++){ //Use C++ in C :..)
		if(sol[c] == row) return 0;
		if(sol[c] - c == row - col) return 0;
		if(n - sol[c] - c == n - row - col) return 0;
	}
	return 1;
}

__global__ void solve(int *psols, int *souls, int n, int partials, int start_col){ //Kernel for solving partial solutions
/* psouls is an array of all partial solutions
souls is an array for the number of souls each partial solution yields
n is the size of the board/number of queens needed to be placed
partials is number of partial solutions generated 
start col is how far the board is completed already
*/
	int index = blockIdx.x*blockDim.x + threadIdx.x;   //get thread ID
	if(index >= partials) return;                      //limit which threads run

	int row = 0, col = start_col;
	int temp = 0;          //solution counter
	souls[index] = 0;         //array of all number of solutions
	int sol[N];               //array to store partial solution
	for(int i = 0; i < n; i++) sol[i] = psols[index*n + i];   //copy partial solution into a seperate array

	while(1){
		if(d_is_valid(sol, n, row, col)){              //check if queen can be placed
			sol[col] = row;                         //place queen in that row/col
			row = 0;                                //set row back to 0 and go to the next column
			col++;
			if(col == n){                        //if its a solution then increment solution counter then continue
				temp++;
				row++;
			}
		} else {
			row++;            //if queen cant be placed in that row, try the next row 
		}
		if(row >= n){               //if no more rows to try, backtrack 
			sol[col] = -1;              //set current col back to empty
			col--;                     //go to previous column
			row = sol[col] + 1;          //try  the next row 
		}
		if(col < start_col) break;        //exit condition 
	}
	souls[index] = temp;         //store number of solution this thread found
	__syncthreads();
}

int generate_partial_solutions(int* psouls, int depth, int n, int threads){ //Function used to generate partial solutions
	/*psouls is the array to store partial solutions in 
	depth is how many columns you want to generate the partial solutions up to
	n is length of board/number of queens
	threads is unused variable from previous implementation
	works in a similar way to the serial iterative solution, except the exit criteria is at the depth and not the last column

	*/


	// Start at first partial solution
	int psi = 0;
	int sol[n];
	int row = 0, col = 0;

	for(int i = 0; i < n; i++) sol[i] = -1; // initialize partial solution to -1

	while(1){
		if(is_valid(sol, n, row, col)){ //check if a queen can be placed
			sol[col] = row;         //if it can place it in that row/column
			row = 0;                //reset row
			col++;                   //go to next column
			if(col == depth){         //if it is a partial solution, store the current board configuration as a partial solution
				for(int i = 0; i < n; i++)
					psouls[psi*n + i] = sol[i];
				psi++; //increment counter for number of solutions
				row++;    //continue
			}
		} else {
			row++;   //else try different row
		}
		if(row >= n){             //backtrack condition
			sol[col--] = -1;
			row = sol[col] + 1;
		}
		if(col == 0 && row >= n) break;   //exit condition 
	}
	return psi;
}

int solve_partial_sols(int* psouls, int start_col, int n, int partials){ //serial version to solve partial solutions
/* psouls is array of partial solutions
start_col is the column where the partial solution ends
n is length of board/ number of queens required
partials is the  number of partial solutions generated
*/


	int solutions = 0;  //solution counter
	int row, col;
	int sol[n] ;

for(int t = 0; t < partials; t++){            //go through each partial solution 
	//printf("T: %d\n", t);
	row = 0;
	col = start_col;
	for(int k = 0; k < n; k++){                                  //copy the partial solution we are dealing with from psouls into sol
		sol[k] = psouls[t*n + k] ;
//		printf("%d ", sol[k]);
	//	printf("partial sol: %d \n", sol[k]) ;
	}
	//printf("\n");

	while(1){
		int valid = is_valid(sol, n, row, col);   //check if queen can be placed
		if(valid){
			sol[col] = row;                  //if it can place queen 
			row = 0;                          //reset row
			col++;                            //increment column to next column
			if(col == n){                     //if solution found
				solutions++;              //increment solution found
				row++;                    //continue
			}
		} else {
			row++;                            //else try a different row
		}
		if(row >= n){                              //backtrack condition, if no more rows to place queen 
			sol[col] = -1;                      //set current row/col to unexplored
			col--;                             //go back to previous col
			row = sol[col] + 1;               //try next avaliable row
		}
		if(col < start_col) break;             //exit condition, backtracked to start
	}
}
	return solutions;                            //return totoal number of solutins found
}

int main(int argc, char *argv[]){
	int *ps;                                           //partial solutions array
	int depth = atoi(argv[1]);                          //input how many columns should be explored in generating partial solutions
	int n = N;                                          //set size of board
	int threads;                                       //upperbound on threads required
	int num_souls;                                      //number of solutions
	long numerator = factorial((long)n);
	long denominator = factorial((long)n - (long)depth);
	threads = numerator / denominator;                     //calculate upper bound of threads required/partial solutions generated


	ps = (int*) malloc(sizeof(int) * threads * n);        //allocate upper bound of memory for number of partials generated
 
	cudaEvent_t start, end, solve_start;               //timer, also times generating partial solutions
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	int partials = generate_partial_solutions(ps, depth, n, threads);  //generate partial solutions in serial 

	/*
	for(int i = 0; i < partials; i++){
		for(int j = 0; j < n; j++)
			printf("%d ", ps[i*n + j]);
		printf("\n");
	}
	*/


//	num_souls = solve_partial_sols(ps,depth,n, partials) ;
//	printf("Main\n");
//	printf("Number of souls: %d \n", num_souls) ;

	/*
	for(int i = 0; i < partials; i++){
		for(int j = 0; j < n; j++)
			printf("%d ", ps[i*n + j]);
		printf("\n");
	}
	*/


	int size = sizeof(int)*partials*n;                 
	int block_threads = 1024;
	int blocks = ceil(partials / block_threads);   //calculate block/grid size to pass kernel 
	if(blocks == 0) blocks++;                      //incase rounding error

	int *d_ps;                       //partial solution array for device
	cudaEventCreate(&solve_start);        //start timer for only the kernel time
	cudaEventRecord(solve_start, 0);
	checkCudaErrors(cudaMalloc((int**)&d_ps, size));
	checkCudaErrors(cudaMemcpy(d_ps, ps, size, cudaMemcpyHostToDevice));  //malloc partial solution array for device

	int *no_sols = (int*) malloc(sizeof(int) * partials);           //malloc array of solutions.               
	int *d_no_sols;
	checkCudaErrors(cudaMalloc((int**)&d_no_sols, sizeof(int) * partials));
	checkCudaErrors(cudaMemcpy(d_no_sols, no_sols, sizeof(int) * partials, cudaMemcpyHostToDevice));

	solve<<< blocks, block_threads >>>(d_ps, d_no_sols, n, partials, depth);    //run kernel 

	cudaEventRecord(end, 0);  //just timer things
	cudaEventSynchronize(end);
	float time = 0;
	float solve_time = 0;
	cudaEventElapsedTime(&time, start, end);
	cudaEventElapsedTime(&solve_time, solve_start, end);
	printf("Size: %d depth: %d ", n, depth);
	printf("Partial_solution: %d ", partials);
	printf("Total_time: %.6f ", time/1000.0);
	printf("Solve_time: %.6f\n", solve_time/1000.0);

	checkCudaErrors(cudaMemcpy(no_sols, d_no_sols, sizeof(int) * partials, cudaMemcpyDeviceToHost)); //copy data back from device

	//printf("Threads: %d\n", threads);
	int no_solutions = 0;
	for(int i = 0; i < partials; i++){
		if(no_sols[i] > -1){
//			printf("%d ", no_sols[i]);
			no_solutions += no_sols[i];  //sum up number of solutions in serial 
		}
	}
	//Free arrays
	checkCudaErrors(cudaFree(d_ps));
	free(ps);

	return(0);
}
