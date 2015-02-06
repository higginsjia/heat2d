#include <stdio.h>
#include <lcutil.h>
#include <timestamp.h>
#include <string.h>
#include "lib.h"

__global__ void heat(float* A, float* B, int matrixSize, int matrixSize2) {
	
	/*Compute row and column in the matrix*/
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*Translate the 2d indexes in 1d index*/
	int i = row + col * matrixSize;
	
	/*Make the operations if needed*/
	if ((row != 0) && (row != matrixSize - 1) &&
	(col != 0) && (col != matrixSize -1) &&
	i < matrixSize2) {
		B[i] = A[i] +
		0.1 * (A[i - 1] + A[i + 1] - 2 * A[i]) +
		0.1 * (A[i - matrixSize] + A[i + matrixSize] - 2 * A[i]);
	}
}

extern "C" float heat2DGPU(float** A, float** B, int matrixSize, int steps, int threads) {
	
	int i;
	
	/*Cuda matrices creation and initialization*/
	float *Aa, *Bb;
	CUDA_SAFE_CALL( cudaMalloc((void**)&Aa, matrixSize*matrixSize*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&Bb, matrixSize*matrixSize*sizeof(float)) );
	
	
	/*Copy data to device memory*/
	for (i = 0; i < matrixSize; i++) {
		CUDA_SAFE_CALL( cudaMemcpy(&Aa[i * matrixSize], A[i], matrixSize*sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy(&Bb[i * matrixSize], B[i], matrixSize*sizeof(float), cudaMemcpyHostToDevice) );
	}
	
	/*Timer*/
	timestamp t_start;
	
	/*Create Cuda grid*/
	dim3 dimBl(threads, threads);
	dim3 dimGr(FRACTION_CEILING(matrixSize, threads), FRACTION_CEILING(matrixSize, threads));
	
	/*Start the timer*/
	t_start = getTimestamp();
	
	
	/*Make the simulation*/
	for (i = 0; i < steps; i++) {
		if (i % 2 == 0) {
			/*From Aa to Bb*/
			heat<<<dimGr, dimBl>>>(Aa, Bb, matrixSize, matrixSize * matrixSize);
		}
		else {
			/*From Bb to Aa*/
			heat<<<dimGr, dimBl>>>(Bb, Aa, matrixSize, matrixSize * matrixSize);
		}
		/*synchronize threads*/
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
	
	/*Stop the timer*/
	float msecs = getElapsedtime(t_start);
	
	/*Copy data from device memory*/
	for (i = 0; i < matrixSize; i++) {
		CUDA_SAFE_CALL( cudaMemcpy(A[i], &Aa[i * matrixSize], matrixSize*sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaMemcpy(B[i], &Bb[i * matrixSize], matrixSize*sizeof(float), cudaMemcpyDeviceToHost) );
	}
	
	
	/*Clean up*/
	CUDA_SAFE_CALL( cudaFree(Aa) );
	CUDA_SAFE_CALL( cudaFree(Bb) );
	return msecs;
}