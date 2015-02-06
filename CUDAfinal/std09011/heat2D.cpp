#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../std09011/lib.h"


/*Same functionality as the given function*/
void initDat(float** A, float** B, int matrixSize) {
	int i, j;

	for (i = 0; i < matrixSize; i++) {
		for (j = 0; j < matrixSize; j++) {
			A[i][j] = (float)i*j;
			A[i][j] *= (float)((matrixSize - i - 1) * (matrixSize - j - 1));
		}
	}
}

int main(int argc, char* argv[]){
	
	int i, threads, matrixSize, steps;
	float ** A, ** B, msecs;
	
	/*Argument reading and checking*/
	if (argc != 4) {
		printf("Check your arguments.Must be 4.\n");
		return -1;
	}
	
	sscanf(argv[1], "%d", &matrixSize);
	sscanf(argv[2], "%d", &steps);
	sscanf(argv[3], "%d", &threads);
	
	
	/*Arguments must be positive*/
	if (matrixSize <= 0 || steps <= 0 || threads <= 0) {
		printf("Wrong arguments. Must be positive.\n");
		return -1;
	}
	
	/*matrixSize must be dividable with threads*/
	if (matrixSize % threads != 0) {
		printf("matrixSize must be dividable with threads.\n");
		return -1;
	}
	
	/*Threads must be dividable with 16*/
	if (threads % 16 != 0) {
		printf("Threads must be dividable with 32.\n");
		return -1;
	}

	
	/*Create and initialize the matrices*/
	A = (float**)malloc(sizeof(float*) * matrixSize);
	if (A == NULL) {
		printf("malloc failed for A.\n");
		return -1;
	}
	
	B = (float**)malloc(sizeof(float*) * matrixSize);
	if (B == NULL) {
		printf("malloc failed for B.\n");
		return -1;
	}
	
	
	for (i = 0; i < matrixSize; i++) {
		A[i] = (float*)malloc(sizeof(float) * matrixSize);
		if (A[i] == NULL) {
			printf("malloc failed for A[%d].\n", i);
			return -1;
		}
		B[i] = (float*)malloc(sizeof(float) * matrixSize);
		if (B[i] == NULL) {
			printf("malloc failed for B[%d].\n", i);
			return -1;
		}
	}
	

	initDat(A, B, matrixSize);
	

	/*Make the simulation and get the time*/
	msecs = heat2DGPU(A, B, matrixSize, steps, threads);
	
	
	/*Clean up*/
	for (i = 0; i < matrixSize; i++) {
		free(A[i]);
		free(B[i]);
	}
	free(A);
	free(B);
	
	/*Print Statistics*/
	printf("threads : %d\n", threads);
	printf("matrix : %d x %d\n", matrixSize, matrixSize);
	printf("cells : %lld\n", (long long int)(matrixSize * matrixSize));
	printf("Execution time %.2f msecs\n\n", msecs);
	printf("----------------END--------------\n");
	return 0;
}