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

int save_data(const char *filename, int *souls, double *times, int n){
	FILE *f;
	f = fopen(filename, "w");
	for(int i = 0; i < n; i++){
		fprintf(f, "%d:%f:%d\n", i, times[i], souls[i]);
	}
	fclose(f);
	return 1;
}

int is_valid(int* sol, int n, int row, int col){
	if(row >= n) return 0;
	for(int c = 0; c <= col; c++){ //Use C++ in C :..)
		if(sol[c] == row) return 0;
		if(sol[c] - c == row - col) return 0;
		if(n - sol[c] - c == n - row - col) return 0;
	}
	return 1;
}

long factorial(long num){
	if(num == 0) return 1;
	return num*factorial(num - 1);
}

__device__ int d_is_valid(int* sol, int n, int row, int col){
	if(row >= n) return 0;
	for(int c = 0; c <= col; c++){ //Use C++ in C :..)
		if(sol[c] == row) return 0;
		if(sol[c] - c == row - col) return 0;
		if(n - sol[c] - c == n - row - col) return 0;
	}
	return 1;
}

__global__ void solve(int *psols, int *souls, int n, int partials, int start_col){

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index >= partials) return;

	int row = 0, col = start_col;
	int temp = 0;
	souls[index] = 0;
	int sol[N];
	for(int i = 0; i < n; i++) sol[i] = psols[index*n + i];

	while(1){
		if(d_is_valid(sol, n, row, col)){
			sol[col] = row;
			row = 0;
			col++;
			if(col == n){
				temp++;
				row++;
			}
		} else {
			row++;
		}
		if(row >= n){
			sol[col] = -1;
			col--;
			row = sol[col] + 1;
		}
		if(col < start_col) break;
	}
	souls[index] = temp;
	__syncthreads();
}

int generate_partial_solutions(int* psouls, int depth, int n, int threads){
	/*
	long numerator = factorial((long)n);
	long denominator = factorial((long)n - (long)depth);
	*threads = numerator / denominator;
	*/

	// Initialise partial solution array
	//psouls = (int**)malloc(sizeof(int*) * (threads));

	// Start at first partial solution
	int psi = 0;
	int sol[n];
	int row = 0, col = 0;

	for(int i = 0; i < n; i++) sol[i] = -1;

	while(1){
		//printf("Col: %d row:  %d\n", col, row);
		if(is_valid(sol, n, row, col)){
			sol[col] = row;
			row = 0;
			col++;
			if(col == depth){
				for(int i = 0; i < n; i++)
					psouls[psi*n + i] = sol[i];
				psi++;
				row++;
			}
		} else {
			row++;
		}
		if(row >= n){
			sol[col--] = -1;
			row = sol[col] + 1;
		}
		if(col == 0 && row >= n) break;
	}
	return psi;
}

int solve_partial_sols(int* psouls, int start_col, int n, int partials){

	int solutions = 0;
	int row, col;
	int sol[n] ;

for(int t = 0; t < partials; t++){
	printf("T: %d\n", t);
	row = 0;
	col = start_col;
	for(int k = 0; k < n; k++){
		sol[k] = psouls[t*n + k] ;
//		printf("%d ", sol[k]);
	//	printf("partial sol: %d \n", sol[k]) ;
	}
	printf("\n");

	while(1){
		int valid = is_valid(sol, n, row, col);
		if(valid){
			sol[col] = row;
			row = 0;
			col++;
			if(col == n){
				solutions++;
				row++;
			}
		} else {
			row++;
		}
		if(row >= n){
			sol[col] = -1;
			col--;
			row = sol[col] + 1;
		}
		if(col < start_col) break;
	}
}
	return solutions;
}

int main(int argc, char *argv[]){
	int *ps;
	int depth = atoi(argv[1]);
	int n = N;
	int threads;
	int num_souls;
	long numerator = factorial((long)n);
	long denominator = factorial((long)n - (long)depth);
	threads = numerator / denominator;


	ps = (int*) malloc(sizeof(int) * threads * n);

	cudaEvent_t start, end, solve_start;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	int partials = generate_partial_solutions(ps, depth, n, threads);

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
	int blocks = ceil(partials / block_threads);
	if(blocks == 0) blocks++;

	int *d_ps;
	cudaEventCreate(&solve_start);
	cudaEventRecord(solve_start, 0);
	checkCudaErrors(cudaMalloc((int**)&d_ps, size));
	checkCudaErrors(cudaMemcpy(d_ps, ps, size, cudaMemcpyHostToDevice));

	int *no_sols = (int*) malloc(sizeof(int) * partials);
	int *d_no_sols;
	checkCudaErrors(cudaMalloc((int**)&d_no_sols, sizeof(int) * partials));
	checkCudaErrors(cudaMemcpy(d_no_sols, no_sols, sizeof(int) * partials, cudaMemcpyHostToDevice));

	solve<<< blocks, block_threads >>>(d_ps, d_no_sols, n, partials, depth);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float time = 0;
	float solve_time = 0;
	cudaEventElapsedTime(&time, start, end);
	cudaEventElapsedTime(&solve_time, solve_start, end);
	printf("Size: %d depth: %d ", n, depth);
	printf("Partial_solution: %d ", partials);
	printf("Total_time: %.6f ", time/1000.0);
	printf("Solve_time: %.6f\n", solve_time/1000.0);

	checkCudaErrors(cudaMemcpy(no_sols, d_no_sols, sizeof(int) * partials, cudaMemcpyDeviceToHost));

	//printf("Threads: %d\n", threads);
	int no_solutions = 0;
	for(int i = 0; i < partials; i++){
		if(no_sols[i] > -1){
//			printf("%d ", no_sols[i]);
			no_solutions += no_sols[i];
		}
	}
	//Free arrays
	checkCudaErrors(cudaFree(d_ps));
	free(ps);

	return(0);
}
