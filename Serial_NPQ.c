#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int is_valid(int* part_sol, int n, int row, int col){
	for(int c = 0; c <= col; c++){ // Let's use C++ in C ...
		if(part_sol[c] == row) return 0;
		if(part_sol[c] - c == row - col) return 0;
		if(n - part_sol[c] - c == n - row - col) return 0;
	}
	return 1;
} 

void print_arr(int *sol, int n){
	for(int i = 0; i < n; i++)
		printf("%d ", sol[i]);
	printf("\n");
}

int solve(int N){
	int sol[N];
	int solutions = 0;
	int row = 0;

	for(int i = 0; i < N; i++) sol[i] = -1; // Initialise to -1

	for(int col = 0; col < N; col++){
		print_arr(sol, N);
		if(!is_valid(sol, N, row, col)){
			row++;
			col--;
			continue;
		} 
		if(col == N - 1){
			solutions++;
			row++;
			col--;
			continue;
		}
		sol[col] = row;
		row = 0;
	}
	printf("Solutions: %d\n", solutions);
}

int main(int argc, char **argv){
	/*
	int test[4] = {1, -1, -1, -1};
	if(is_valid(test, 4, 2, 3))
		printf("valid\n");
	else
		printf("invalid\n");
	*/

	int N = atoi(argv[1]);
	solve(N);
	return 0;
}
