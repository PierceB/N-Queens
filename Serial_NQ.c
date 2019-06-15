// N-Queens Serial implementation (Pierce Burke and Zachary Bowditch)

#include <stdio.h>
#include <stdlib.h>

int N = 8;
int no_solutions = 0; // increment every time a solution is found 


int isValidMove(int board[N][N], int row,int col){
	//Returns 0 if not valid or 1 if valid
  int i, j; 
	//Check row
	for(int i = 0; i<N;i++){             //check all blocks in the same row
		if(i != col){                    //exclude the current queen we jst placed
			if(board[row][i])
				return(0);
		}
	}
//	check column
	for(int i = 0; i<N;i++){
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
		for(i = row, j = col; i<N && j >= 0; i++,j--){
				if(board[i][j])
					return(0);
			}
		// upper right diagonal
		for(i = row, j = col; i>=0 && j < N; i--,j++){
					if(board[i][j])
						return(0);
				}
		//lower right diagonal
		for(i = row, j = col; i< N && j < N; i++,j++){
					if(board[i][j])
						return(0);
				}
		return(1);
		
}

int solveBoard(int board[N][N], int col){
	//recursive implementation for solving N-Queens
	
	if(col >= N)
		return(1); //Exit condition for recursion

	for(int i = 0 ; i< N; i++){            //try a queen in each row for the current col
		if(isValidMove(board,i,col)){
			board[i][col] = 1;              //if its allowed put a queen there
			
			if(solveBoard(board,col+1))      //call the solve method with the queen in this position
				no_solutions++;
				//return(1) ;                //if it can find a solution then return 1. (Return first solution found)
						
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

int main(int argc, char *argv[]){
	
	if(argc > 1)
		N =  atoi(argv[1]);    //If a value is supplied when run. Use that as N 

	int board[N][N] ;  //create board 
	
	for(int i = 0 ; i < N ; i++)
		for(int k = 0 ; k < N ; k++){
			board[i][k] = 0 ;  //Board initialization 
		}
	
	printBoard(board); //print blank board
	
	if(solveBoard(board,0) == 0) //If no solution is found 
		printf("No. of solutions: %d\n", no_solutions);
	else {
		printf("Solution=======\n");
		printBoard(board);   //print finished board
	}
	
	return(0);
}
