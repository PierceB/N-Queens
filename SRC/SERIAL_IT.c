#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int is_valid(int* sol, int n, int row, int col){ // helper function for checking if a queen can be placed 
	if(row >= n) return 0; //check if on board
	for(int c = 0; c <= col; c++){ 
		if(sol[c] == row) return 0; //check row
		if(sol[c] - c == row - col) return 0; //check main diagonal
		if(n - sol[c] - c == n - row - col) return 0; // check off diagonal
	}
	return 1;
}

void print_arr(int *sol, int n){ //helper function to print out the array
	for(int i = 0; i < n; i++)
		printf("%d ", sol[i]);
	printf("\n");
}

int solve(int N){
	int sol[N];
	int solutions = 0;
	int row = 0, col = 0;

	for(int i = 0; i < N; i++) sol[i] = -1; //initalize array to everyhing unexplored

	while(1){
		if(is_valid(sol, N, row, col)){ //Check if queen can be placed in current position 
			sol[col] = row; //if it can place it, reset row to 0 and go to next column 
			row = 0; 
			col++;
			if(col == N){ //if solution found increment nmber of solutions and continue
				solutions++; 
				row++;
			}
		} else { //else continue
			row++;
		}
		if(row >= N){ //if in no more rows, set current value to unexplored and backtrack
			sol[col--] = -1;
			row = sol[col] + 1;
		}
		if(col == 0 && row >= N) break; //if in first column and no more rows, end
	}
	return solutions;
}

int main(int argc, char **argv){
	for(int n = 1; n <= 8; n++){ //run the above for each sized board from 1-15 
		clock_t start = clock();		
		int souls = solve(n);
		clock_t end = clock();
		double duration = (double) (end - start) / CLOCKS_PER_SEC;
		printf("n: %d sols: %d time: %.6f\n", n, souls, duration);
	}
	return 1;
}
