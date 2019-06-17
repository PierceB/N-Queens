// CUDA approach for solving the N-Queens problem (Pierce Burke and Zachary Bowditch)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string.h>


// include CUDA files
#include <cuda_runtime.h>

#include "inc/helper_functions.h"
#include "inc/helper_cuda.h"

int N = 8;
int no_solutions; // increment every time a solution is found


__global__ void isValidMoveGPU(int *board, int row, int col, int n, int isValid){
  unsigned int icol = blockIdx.x*blockDim.x + threadIdx.x;              //find x dimension
  unsigned int irow = blockIdx.y*blockDim.y + threadIdx.y ;
if((irow != row) || (icol != col)){

  if(irow == row){
    if((irow < n) && (icol < n)){
      if(board[irow*n + icol]){
        isValid += 1 ;
      }
    }
  }

  if(icol == col){
    if((irow < n) && (icol < n)){
      if(board[irow*n + icol]){
        isValid += 1 ;
        return;
      }
    }
  }

  if((irow-icol) == (row-col)){
    if((irow < n) && (icol < n)){
      if(board[irow*n + icol]){
        isValid += 1 ;
        return ;
      }
    }
  }

  if((n-irow-icol) == (n-row-col)){
    if((irow < n) && (icol < n)){
      if(board[irow*n + icol]){
        isValid += 1 ;
        return ;
      }
    }
  }

  }

}




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

int solveBoard(int **board, int col, int n){
	//recursive implementation for solving N-Queens
  int disValid;
  int hisValid,t;
  hisValid = 0 ;
  int *dBoard = 0 ;

  cudaMalloc((void**)&dBoard, n*n*sizeof(int)) ;
  //cudaMalloc(disValid, sizeof(int));






  size_t block_size = 64 ;
  size_t grid_size = ceil(n*n / block_size) ;

	if(col >= n)
		return(1); //Exit condition for recursion (check within board dimensions)

	for(int i = 0 ; i < n; i++){            //try a queen in each row for the current col
    cudaMemcpy(dBoard,board,n*n*sizeof(int),cudaMemcpyHostToDevice);
    isValidMoveGPU<<<grid_size, block_size>>>(dBoard,i,col,n,disValid);
    cudaMemcpy(&hisValid, &disValid,sizeof(int),cudaMemcpyDeviceToHost);
    printf("%d \n", hisValid);
    if(hisValid > 0){
			board[i][col] = 1;              //if its allowed put a queen there

			if(solveBoard(board,col+1, n)){      //call the solve method with the queen in this position
				no_solutions++;

				//return(1) ;                //if it can find a solution then return 1. (Return first solution found)
			}
      t = 0 ;
      cudaMemcpy(&disValid,&t,sizeof(int),cudaMemcpyHostToDevice) ;
			board[i][col] = 0;          //Else backtrack.
			}
		}
	return(0);                   //returns if no solution found.
}


// void printBoard(int board[N][N]){
// 	//Method to print the board
// 	for(int i = 0 ; i < N ; i++){
// 			for(int k = 0 ; k < N ; k++){
// 				printf("%d ",board[i][k]) ;
// 			}
// 			printf("\n");
// 	}
// }

int save_data(const char *filename, int *souls, double *times, int N){
	FILE *f;
	f = fopen(filename, "w");
	for(int i = 0; i < N; i++){
		fprintf(f, "%d:%f:%d\n", i, times[i], souls[i]);
	}
	fclose(f);
	return 1;
}

int main(int argc, char *argv[]){

	int averages;





	if(argc > 2){
		N =  atoi(argv[1]);    //If a value is supplied when run. Use that as N
		averages = atoi(argv[2]);
	} else {
		printf("Usage: %s [N (up to)] [number of averages] [filename]\n", argv[0]);
		exit(1);
	}

	double sum;
	double times[N];
	double total;
	int souls[N];



	for(int n = 1; n <= N; n++){
		no_solutions = 0;
		sum = 0.0;
		for(int count = 0; count < averages; count++){

			int **board = (int**)malloc(sizeof(int*) * n);
			for(int i = 0 ; i < n ; i++){
				board[i] = (int*)malloc(sizeof(int) * n);
				for(int k = 0 ; k < n ; k++)
					board[i][k] = 0 ;  //Board initialization
			}

		//	printBoard(board); //print blank board

			clock_t start = clock();
			solveBoard(board, 0, n); //If no solution is found
//			printf("No. of solutions: %d\n", no_solutions);
			printf("Solutions for %d: %d\n", n, no_solutions);

			clock_t end = clock();
			total = (double)(end - start) / CLOCKS_PER_SEC;

			sum += total;
		}
		times[n] = sum / averages;
		souls[n] = no_solutions;
	}

	if(argc > 3) save_data(argv[3], souls, times, N);

	return(0);
}
