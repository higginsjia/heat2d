#ifndef _LIB_H_
#define _LIB_H_


extern "C" float heat2DGPU(float** A, float** B, int matrixSize, int steps, int threads);

#endif