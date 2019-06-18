
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

int N = 8;
long souls;

int isValidMove(int **board, int row,int col, int n){
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

int solveBoard(int **board, int count, int col, int n){
	//recursive implementation for solving N-Queens

	if(count >= n)	// if(argc > 2){
	// 	N =  atoi(argv[1]);    //If a value is supplied when run. Use that as N
	// 	averages = atoi(argv[2]);
	// } else {
	// 	printf("Usage: %s [N (up to)] [number of averages] [filename]\n", argv[0]);
	// 	exit(1);
	// }
		return(1); //Exit condition for recursion (check within board dimensions)

	for(int i = 1 ; i < n; i++){            //try a queen in each row for the current col
		if(isValidMove(board,i,col, n)){
			board[i][col] = 1;              //if its allowed put a queen there

			if(solveBoard(board, count + 1, (col+1) % n, n)){      //call the solve method with the queen in this position
				souls++;
				//return(1) ;                //if it can find a solution then return 1. (Return first solution found)
			}

			board[i][col] = 0;          //Else backtrack.
			}
		}
	return(0);                   //returns if no solution found.
}

void printBoard(int board[N][N]){
	//Method to print the board
	for(int i = 0 ; i < N ; i++){
			for(int k = 0 ; k < N ; k++){
				printf("%d ",board[i][k]) ;
			}
			printf("\n");
	}
}

int save_data(const char *filename, long *souls, double *times, int N){
	FILE *f;
	f = fopen(filename, "w");

	for(int i = 0; i < N; i++)
		fprintf(f, "%d:%f:%ld\n", i, times[i], souls[i]);

	fclose(f);
	return 1;
}

int main(int argc, char *argv[]){

	int averages;

	// if(argc > 2){
	// 	N =  atoi(argv[1]);    //If a value is supplied when run. Use that as N
	// 	averages = atoi(argv[2]);
	// } else {
	// 	printf("Usage: %s [N (up to)] [number of averages] [filename]\n", argv[0]);
	// 	exit(1);
	// }

	double sum;
	double times[N];
	double total;
  long totalSouls;
   int n = N;
  int procs, rank;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int stride = n/(procs) ;
  int finalNum = (procs - 1)*stride  ;

	if(rank < N){

		int r = rank;
		int **board = malloc(sizeof(double*) * n);
		for(int i = 0 ; i < n ; i++){
			board[i] = malloc(sizeof(double) * n);
			for(int k = 0 ; k < n ; k++)
				board[i][k] = 0 ;  //Board initialization
		}

		while(1){
			board[0][r] = 1;
			solveBoard(board, 1, (r + 1) % n , n);

			for(int i = 0 ; i < n ; i++)
				for(int k = 0 ; k < n ; k++)
					board[i][k] = 0 ;  //Board initialization


			r += procs;
			if(r >= N) break;
		}
	} else {
		souls = 0;
	}

	  MPI_Reduce(&souls, &totalSouls,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);


      if(rank == 0){
        printf("Number of solutions for %d: %ld \n",n, totalSouls);
      }

  MPI_Finalize();

	return(0);
}
