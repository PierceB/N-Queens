// N-Queens MPI implementation (Pierce Burke and Zachary Bowditch)

#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

// Check board configuration for validity
int is_valid_move(int **board, int row, int col, int n){
	//Returns 0 if not valid or 1 if valid
  int i, j;
	//Check row
	for(int i = 0; i<n;i++){             //check all blocks in the same row
		if(i != col){                    //exclude the current queen we jst placed
			if(board[row][i])
				return(0);
		}
	}
//	check column
	for(int i = 0; i<n;i++){
			if(i != row){
				if(board[i][col])
					return(0) ;
			}
		}
	//Check diagonals
		//upper left diagonal

		for(i = row, j = col; i>=0 && j >= 0; i--,j--){
			if(board[i][j])
				return(0);
		}
		//lower left diagonal
		for(i = row, j = col; i<n && j >= 0; i++,j--){
				if(board[i][j])
					return(0);
			}
		// upper right diagonal
		for(i = row, j = col; i>=0 && j < n; i--,j++){
					if(board[i][j])
						return(0);
				}
		//lower right diagonal
		for(i = row, j = col; i< n && j < n; i++,j++){
					if(board[i][j])
						return(0);
				}
		return(1);

}

long solve_board(int **board, int col, int n){
	//recursive implementation for solving N-Queens

	long no_solutions = 0;

	if(col >= n)
		return(1); //Exit condition for recursion (check within board dimensions)

	for(int i = 0 ; i < n; i++){            //try a queen in each row for the current col
		if(is_valid_move(board, i, col, n)){
			board[i][col] = 1;              //if its allowed put a queen there

			if(solve_board(board,col+1, n)){      //call the solve method with the queen in this position
				no_solutions++;
				//return(1) ;                //if it can find a solution then return 1. (Return first solution found)
			}

			board[i][col] = 0;          //Else backtrack.
			}
		}
	return no_solutions;
}

long find_solutions(int n){
	long solutions = 0;
	int procs, rank;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*
	int rem = N % procs;
	int cands = N / (procs - rem);

	if(cands >= 2 * N){

	}

	int col;
	//if(rank < n + 1) col = rank;
	*/

	int **board = (int**)malloc(sizeof(int*) * n);
	for(int i = 0; i < n; i++)
		board[i] = (int*)calloc(n, sizeof(int));

	board[0][rank] = 1;

	long souls = solve_board(board, 1, n);

	MPI_Reduce(&souls, &solutions, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD) ;
	MPI_Finalize();

	if(rank == 0){
		printf("Solutions found: %ld\n", solutions);
		return solutions;
	}

	return 0;
}

void get_board_solutions(int N){
	long *board_solutions = malloc(sizeof(long)*N);
	double *times = malloc(sizeof(double)*N);

	for(int n = 0; n < 1; n++){
		//start timer
		clock_t start = clock();
		board_solutions[n] = find_solutions(n);
		//end_timer
		clock_t end = clock();
		times[n] = (double)(end - start);
	}
	for(int i = 0; i < 1; i++)
		printf("solutions[%d] = %ld\n", i, board_solutions[i]);
}

int main(int argc, char **argv){
	get_board_solutions(5);
	return 0;
}
