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

#define N 4 

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

__device__ int d_solutions;

/*
__global__ void test(int *a){
	a[0] = 1;
	printf("test21 \n"); 
	a[1] = 2;
}
*/

__global__ void solve(int *psols, int *souls, int n, int partials, int start_col){

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index >= partials) return;

	int row = 0, col = start_col;
	souls[index] = 0;
	int sol[N];
	for(int i = 0; i < n; i++) sol[i] = psols[index*n + i];

	while(1){
		if(d_is_valid(sol, n, row, col)){
			sol[col] = row;
			row = 0;
			col++;
			if(col == n){
				souls[index]++;
				row++;
			}
		} else {
			row++;
		}
		if(row >= n){
			sol[col--] = -1;
			row = sol[col] + 1;
		}
		if(col == start_col && row >= n) break;
	}
	__syncthreads();
}

int generate_partial_solutions(int** psouls, int depth, int n, int threads){
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
				psouls[psi] = (int*)malloc(sizeof(int) * n);
				for(int i = 0; i < n; i++)
					psouls[psi][i] = sol[i];
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

int solve_partial_sols(int** psouls,int start_col, int n){

	int solutions = 0;
	int row, col = start_col;
	int sol[n] ;

for(int t = 0; t < n; t++){
	row = 0;
	for(int k = 0; k < n; k++){
		sol[k] = psouls[t][k] ;
	//	printf("partial sol: %d \n", sol[k]) ;
	}

	while(1){
		if(is_valid(sol, n, row, col)){
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
			sol[col--] = -1;
			row = sol[col] + 1;
		}
		if(col == start_col && row >= n) break;
	}
}
	return solutions;
}

int main(int argc, char *argv[]){
	int **ps;
	int depth = 1;
	int n = N;
	int threads;
	long numerator = factorial((long)n);
	long denominator = factorial((long)n - (long)depth);
	threads = numerator / denominator;

	ps = (int**)malloc(sizeof(int) * threads * n);

	printf("Threads: %ld\n", threads);
	int partials = generate_partial_solutions(ps, depth, n, threads);
	printf("Psols: %d\n", partials);

	/*
	num_souls = solve_partial_sols(ps,depth,n) ;
	printf("Number of souls: %d \n", num_souls) ;

	for(int i = 0; i < partials; i++){
		for(int j = 0; j < n; j++)
			printf("%d ", ps[i][j]);
		printf("\n");
	}
	*/

	int size = sizeof(double)*partials*n;
	int block_threads = 12;
	int blocks = ceil(threads / block_threads);

	int *d_ps;
	checkCudaErrors(cudaMalloc((void**)&d_ps, size));
	checkCudaErrors(cudaMemcpy(d_ps, ps, size, cudaMemcpyHostToDevice));
	void *souls;
	int *no_sols = (int*)malloc(sizeof(int) * threads);
	int *d_no_sols;
	checkCudaErrors(cudaMemcpy(d_no_sols, &no_sols, sizeof(int) * threads, cudaMemcpyHostToDevice));
	cudaGetSymbolAddress(&souls, d_solutions);
	printf("here \n"); 

	solve<<<blocks, block_threads>>>(d_ps, d_no_sols, n, partials, depth);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(no_sols, d_no_sols, sizeof(int) * partials, cudaMemcpyDeviceToHost));

	int no_solutions = 0;
	for(int i = 0; i < partials; i++) no_solutions += no_sols[i];
//	cudaMemcpyFromSymbol(&no_solutions, "d_solutions", sizeof(int), 0, cudaMemcpyDeviceToHost);
	printf("Solutions found: %d\n", no_solutions);
	//Free arrays
	checkCudaErrors(cudaFree(d_ps));
	for(int i = 0; i < partials; i++) free(ps[i]);
	free(ps);

	return(0);
}
