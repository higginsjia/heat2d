#ifndef CART_H
#define CART_H

#define TRUE 1
#define FALSE 0
#define SEND 0
#define RECV 1
#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3
#define MYTAG 321

typedef struct Chart {
	float* t[2];
} Chart;


typedef struct Buffer {
	float* b;
} Buffer;

void innerUpdate(Chart chart, int size, int iz);

void outerUpdate(Chart chart, int size, int iz, Buffer* buffs, int* neighbors);

int pos(int i, int j, int size);

void initDat(Chart chart, int* coords, int dimension, int matrixSize);

int initChart(Chart* chart, int size, int id);

void freeChart(Chart* chart, int size);

int initBuffs(Buffer** buffs, int id);

void freeBuffs(Buffer** buffs);

int initNeighbors(int* neighbors, Buffer* buffs, int size);

void freeNeighbors(int* neighbors, Buffer* buffs);

int initBuff(Buffer* buff, int size);

void freeBuff(Buffer* buff);

void simulation(Buffer* buffs, int* neighbors, Chart chart, int size, int steps, MPI_Comm cartComm,
				MPI_Request* upRequest, MPI_Request* downRequest, MPI_Request* leftRequest, MPI_Request* rightRequest,
				MPI_Status* upStatus, MPI_Status* downStatus, MPI_Status* leftStatus, MPI_Status* rightStatus,
				MPI_Datatype columnType, MPI_Datatype rowType);

#endif