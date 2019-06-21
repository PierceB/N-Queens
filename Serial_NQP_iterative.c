#include <stdio.h>
#include <stdlib.h>

int is_valid(int* sol, int n, int row, int col){
	if(row >= n) return 0;
	for(int c = 0; c <= col; c++){ //Use C++ in C :..)
		if(sol[c] == row) return 0;
		if(sol[c] - c == row - col) return 0;
		if(n - sol[c] - c == n - row - col) return 0;
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
	int row = 0, col = 0;

	for(int i = 0; i < N; i++) sol[i] = -1;

	while(1){
		if(is_valid(sol, N, row, col)){
			sol[col] = row;
			row = 0;
			col++;
			if(col == N){
				solutions++;
				row++;
			}
		} else {
			row++;
		}
		if(row >= N){
			sol[col--] = -1;
			row = sol[col] + 1;
		}
		if(col == 0 && row >= N) break;
	}
	return solutions;
}

int main(int argc, char **argv){
	int N = 8;
	int souls = solve(N);
	printf("Solutions: %d\n", souls);
	return 1;
}
