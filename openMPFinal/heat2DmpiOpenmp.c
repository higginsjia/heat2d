#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "heat2DmpiOpenmp.h"


struct Parms { 
	float cx;
	float cy;
} parms = {0.1, 0.1};


int main(int argc, char* argv[]) {
	
	int numTasks, procs, matrixSize, id, id2d, steps, neighbors[4], size, threadNum;
	MPI_Comm cartComm;
	/*requests 0 for sending 1 for receiving*/
	MPI_Request leftRequest[2], rightRequest[2], upRequest[2], downRequest[2];
	MPI_Status leftStatus[2], rightStatus[2], upStatus[2], downStatus[2];
	double start, end;
	
	/*MpiDatatype variables*/
	MPI_Datatype columnType, rowType;
	
	/*Topology variables*/
	int dim[2], coords[2], period[2], reorder;
	
	/*buffers for receiving*/
	Buffer* buffs;
	Chart chart; 
	
	
	//mpi calls
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	
	/*Arguments reading and checking*/
	if (argc != 4) {
		printf("Check your arguments.Must be 4.\n");
		return -1;
	}
	
	sscanf(argv[1], "%d", &procs);
	sscanf(argv[2], "%d", &matrixSize);
	sscanf(argv[3], "%d", &steps);
	
	/*procs must be numTasks' square*/
	if (sqrt(numTasks) != procs) {
		printf("Check your processes numbers.\n");
		return -1;
	}
	
	/*420 is least common multiplier of 2,3,4,5,6,7.*/
	if (matrixSize >= 420 && matrixSize%420 != 0) {
		printf("Check your matrixSize number.\n");
		return -1;
	}
	
	/*steps must be even*/
	if (steps % 2 != 0) {
		printf("Check your steps number.\n");
		return -1;
	}
	
	/*Number of threads*/
	threadNum = omp_get_num_threads();
	
	/*initializing buffers*/
	if (initBuffs(&buffs, id) == -1) {
		return -1;
	}
	
	/*Topology and datatype initialization*/
	size = matrixSize / procs;
	
	/*grid variables*/
	dim[0] = dim[1] = procs;
	period[0] = period[1] = 0;
	reorder = 0;
	neighbors[LEFT] = neighbors[RIGHT] = neighbors[UP] = neighbors[DOWN] = -1;
	
	/*column vector*/
	MPI_Aint stride1 = size * sizeof(float);
	MPI_Type_create_hvector(size, 1, stride1, MPI_FLOAT, &columnType);
	MPI_Type_commit(&columnType);
	
	/*row vector*/
	MPI_Aint stride2 = sizeof(float);
	MPI_Type_create_hvector(size, 1, stride2, MPI_FLOAT, &rowType);
	MPI_Type_commit(&rowType);
	
	/*data initialization*/
	if (initChart(&chart, size, id) == -1) {
		return -1;
	}
	
	/*Presentation printing*/
	if (id == 0) {
		printf("\nnumTasks = %d\tsteps = %d\n", numTasks, steps);
		printf("Matrix : %d x %d\n", matrixSize, matrixSize);
		printf("cells : %lld\n", (long long int)matrixSize * matrixSize);
	}
	
	
	/*Topology creation*/
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);
	
	MPI_Comm_rank(cartComm, &id2d);
	MPI_Cart_coords(cartComm, id2d, 2, coords);
	
	/*Topology neighbors*/
	MPI_Cart_shift(cartComm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
	MPI_Cart_shift(cartComm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
	
	/*Buffers initialization*/
	if (initNeighbors(neighbors, buffs, size) == -1) {
		return -1;
	}
	
	
	/*Initializing the data*/
	initDat(chart, coords, size, matrixSize);
	
	/*Start the timer*/
	start = MPI_Wtime();
	
	/*Get the work done.*/
	simulation(buffs, neighbors, chart, size, steps, threadNum, cartComm,
			   upRequest, downRequest, leftRequest, rightRequest,
			upStatus, downStatus, leftStatus, rightStatus,
			columnType, rowType);
	
	/*Stop the timer*/
	end = MPI_Wtime();
	
	
	/*Clean up*/
	freeChart(&chart, size);
	
	freeNeighbors(neighbors, buffs);
	
	freeBuffs(&buffs);
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Type_free(&columnType);
	MPI_Type_free(&rowType);
	MPI_Comm_free(&cartComm);
	MPI_Finalize();
	
	
	/*Print time*/
	if(id == 0) {
		printf("\nDuration is: %f\n", end - start);
	}
	
	return 0;
}


/*Gives the one dimension pos*/
int pos(int i, int j, int size) {
	return (i * size) + j;
}

/*Same functionality as the previous iniDat but works individually*/
void initDat(Chart chart, int* coords, int dimension, int matrixSize) {
	int i, j;
	for (i = 0; i < dimension; i++) {
		for (j = 0; j < dimension; j++) {
			chart.t[0][pos(i, j, dimension)] = (float)((i + coords[0] * dimension) * (j + coords[1] * dimension));
			chart.t[0][pos(i, j, dimension)] *= (float)((matrixSize - (i + coords[0] * dimension) - 1) * (matrixSize - (j + coords[1] * dimension) - 1));
		}
	}
}

int initChart(Chart* chart, int size, int id) {
	
	int i;
	for (i = 0; i < 2; i++) {
		chart->t[i] = malloc(sizeof(float) * size * size);
		if (chart->t[i] == NULL) {
			printf("Malloc failed for chart proc : %d.\n", id);
			return -1;
		}
	}
	return 0;
}

void freeChart(Chart* chart, int size) {
	
	int i;
	for (i = 0; i < 2; i++) {
		free(chart->t[i]);
	}
	
}


int initBuff(Buffer* buff, int size) {
	
	buff->b = malloc(sizeof(float) * size);
	if (buff->b == NULL) {
		printf("initBuff failed.\n");
		return -1;
	}
	return 0;
}

void freeBuff(Buffer* buff) {
	
	free(buff->b);
	buff->b = NULL;
}

int initBuffs(Buffer** buffs, int id) {
	
	(*buffs) = malloc(sizeof(Buffer) * 4);
	if ((*buffs) == NULL) {
		printf("buffs failed to initialize, id = %d.\n", id);
		return -1;
	}
	return 0;
}

void freeBuffs(Buffer** buffs) {
	
	free(*buffs);
	(*buffs) = NULL;
}


int initNeighbors(int* neighbors, Buffer* buffs, int size) {
	
	if (neighbors[UP] != -1) {
		if (initBuff(&buffs[UP], size) == -1){
			return -1;
		}
	}
	if (neighbors[DOWN] != -1) {
		if (initBuff(&buffs[DOWN], size) == -1){
			return -1;
		}
	}
	if (neighbors[LEFT] != -1) {
		if (initBuff(&buffs[LEFT], size) == -1){
			return -1;
		}
	}
	if (neighbors[RIGHT] != -1) {
		if (initBuff(&buffs[RIGHT], size) == -1){
			return -1;
		}
	}
	return 0;
}

void freeNeighbors(int* neighbors, Buffer* buffs) {
	
	if (neighbors[UP] != -1) {
		freeBuff(&buffs[UP]);
	}
	if (neighbors[DOWN] != -1) {
		freeBuff(&buffs[DOWN]);
	}
	if (neighbors[LEFT] != -1) {
		freeBuff(&buffs[LEFT]);
	}
	if (neighbors[RIGHT] != -1) {
		freeBuff(&buffs[RIGHT]);
	}
	
}


/*Update the inner cells of the cart.*/
void innerUpdate(Chart chart, int size, int iz, int threadNum) {
	
	int i, j;
	#pragma omp parallel shared(chart, size, iz, parms) private(i, j) num_threads(threadNum)
	for (i = 1; i < size - 1; i++) {
		for (j = 1; j < size - 1; j++) {
			chart.t[1 - iz][(i * size) + j] = chart.t[iz][(i * size) + j]  + parms.cx * (chart.t[iz][((i + 1) * size) + j] + chart.t[iz][((i - 1) * size) + j] - 2.0 * chart.t[iz][(i * size) + j]) +
			parms.cy * (chart.t[iz][(i * size) + j + 1] + chart.t[iz][(i * size) + j - 1] - 2.0 * chart.t[iz][(i * size) + j]);
		}
	}
}

/*Update the outer cells of the cart.*/
void outerUpdate(Chart chart, int size, int iz, Buffer* buffs, int* neighbors, int threadNum) {
	
	int i, j;
	
	/*Up row*/
	/*Openmp call*/
	#pragma omp parallel shared(chart, size, iz, parms, buffs, neighbors) private(i, j) num_threads(threadNum)
	if (neighbors[UP] != -1) {
		for (j = 1; j < size - 1; j++) {
			chart.t[1 - iz][j] = chart.t[iz][j]  +
			parms.cx * (chart.t[iz][size + j] + buffs[UP].b[j] - 2.0 * chart.t[iz][j]) +
			parms.cy * (chart.t[iz][j + 1] + chart.t[iz][j - 1] - 2.0 * chart.t[iz][j]);
		}
		/*Up left corner*/
		if (neighbors[LEFT] != -1) {
			chart.t[1 - iz][0] = chart.t[iz][0]  +
			parms.cx * (chart.t[iz][size] + buffs[UP].b[0] - 2.0 * chart.t[iz][0]) +
			parms.cy * (chart.t[iz][1] + buffs[LEFT].b[0] - 2.0 * chart.t[iz][0]);
		}
		/*Up right corner*/
		if (neighbors[RIGHT] != -1) {
			chart.t[1 - iz][size - 1] = chart.t[iz][size - 1]  +
			parms.cx * (chart.t[iz][size + size - 1] + buffs[UP].b[size - 1] - 2.0 * chart.t[iz][size - 1]) +
			parms.cy * (buffs[RIGHT].b[0] + chart.t[iz][size - 2] - 2.0 * chart.t[iz][size - 1]);
		}
	}
	
	/*Down row*/
	/*Openmp call*/
	#pragma omp parallel shared(chart, size, iz, parms, buffs, neighbors) private(i, j) num_threads(threadNum)
	if (neighbors[DOWN] != -1) {
		for (j = 1; j < size - 1; j++) {
			chart.t[1 - iz][((size - 1) * size) + j] = chart.t[iz][((size - 1) * size) + j]  +
			parms.cx * (buffs[DOWN].b[j] + chart.t[iz][((size - 2) * size) + j] - 2.0 * chart.t[iz][((size - 1) * size) + j]) +
			parms.cy * (chart.t[iz][((size - 1) * size) + j + 1] +chart.t[iz][((size - 1) * size) + j - 1] - 2.0 * chart.t[iz][((size - 1) * size) + j]);
		}
		/*Down left corner*/
		if (neighbors[LEFT] != -1) {
			chart.t[1 - iz][((size - 1) * size)] = chart.t[iz][((size - 1) * size)]  +
			parms.cx * (buffs[DOWN].b[0] + chart.t[iz][((size - 2) * size)] - 2.0 * chart.t[iz][((size - 1) * size)]) +
			parms.cy * (chart.t[iz][((size - 1) * size) + 1] + buffs[LEFT].b[size - 1] - 2.0 * chart.t[iz][((size - 1) * size)]);
		}
		/*Down right corner*/
		if (neighbors[RIGHT] != -1) {
			chart.t[1 - iz][((size - 1) * size) + size - 1] = chart.t[iz][((size - 1) * size) + size - 1]  +
			parms.cx * (buffs[DOWN].b[size - 1] + chart.t[iz][((size - 2) * size) + size - 1] - 2.0 * chart.t[iz][((size - 1) * size) + size - 1]) +
			parms.cy * (buffs[RIGHT].b[size - 1] + chart.t[iz][((size - 1) * size) + size - 2] - 2.0 * chart.t[iz][((size - 1) * size) + size - 1]);
		}
	}
	
	/*Left column*/
	/*Openmp call*/
	#pragma omp parallel shared(chart, size, iz, parms, buffs, neighbors) private(i, j) num_threads(threadNum)
	if (neighbors[LEFT] != -1) {
		for (i = 1; i < size - 1; i++) {
			chart.t[1 - iz][(i * size)] = chart.t[iz][i * size]  +
			parms.cx * (chart.t[iz][((i + 1) * size)] + chart.t[iz][(i - 1) * size] - 2.0 * chart.t[iz][i * size]) +
			parms.cy * (chart.t[iz][i * size + 1] + buffs[LEFT].b[i] - 2.0 * chart.t[iz][i * size]);
		}
	}
	
	/*Right column*/
	/*Openmp call*/
	#pragma omp parallel shared(chart, size, iz, parms, buffs, neighbors) private(i, j) num_threads(threadNum)
	if (neighbors[RIGHT] != -1) {
		for (i = 1; i < size - 1; i++) {
			chart.t[1 - iz][i * size + size - 1] = chart.t[iz][i * size + size - 1]  +
			parms.cx * (chart.t[iz][(i + 1) * size + size - 1] + chart.t[iz][(i - 1) * size + size - 1] - 2.0 * chart.t[iz][i * size + size - 1]) +
			parms.cy * (buffs[RIGHT].b[i] +chart.t[iz][i * size + size - 2] - 2.0 * chart.t[iz][i * size + size - 1]);
		}
	}
	
} 

/*Simulate the heat2D*/
void simulation(Buffer* buffs, int* neighbors, Chart chart, int size, int steps, int threadNum, MPI_Comm cartComm,
				MPI_Request* upRequest, MPI_Request* downRequest, MPI_Request* leftRequest, MPI_Request* rightRequest,
				MPI_Status* upStatus, MPI_Status* downStatus, MPI_Status* leftStatus, MPI_Status* rightStatus,
				MPI_Datatype columnType, MPI_Datatype rowType) {
	
	int i, iz = 0;
	
	for (i = 0; i < steps; i++) {
		
		/*send data*/
		if (neighbors[UP] != -1) {
			MPI_Isend(chart.t[iz], 1, rowType, neighbors[UP], MYTAG, cartComm, &upRequest[SEND]);
			MPI_Irecv(buffs[UP].b, size, MPI_FLOAT, neighbors[UP], MYTAG, cartComm, &upRequest[RECV]);
		}
		if (neighbors[DOWN] != -1) {
			MPI_Isend(&chart.t[iz][(size - 1) * size], 1, rowType, neighbors[DOWN], MYTAG, cartComm, &downRequest[SEND]);
			MPI_Irecv(buffs[DOWN].b, size, MPI_FLOAT, neighbors[DOWN], MYTAG, cartComm, &downRequest[RECV]);
		}
		if (neighbors[LEFT] != -1) {
			MPI_Isend(chart.t[iz], 1, columnType, neighbors[LEFT], MYTAG, cartComm, &leftRequest[SEND]);
			MPI_Irecv(buffs[LEFT].b, size, MPI_FLOAT, neighbors[LEFT], MYTAG, cartComm, &leftRequest[RECV]);
		}
		if (neighbors[RIGHT] != -1) {
			MPI_Isend(&chart.t[iz][size - 1], 1, columnType, neighbors[RIGHT], MYTAG, cartComm, &rightRequest[SEND]);
			MPI_Irecv(buffs[RIGHT].b, size, MPI_FLOAT, neighbors[RIGHT], MYTAG, cartComm, &rightRequest[RECV]);
		}
		
		/*compute the inner array*/
		innerUpdate(chart, size, iz, threadNum);
		
		/*wait until data are available*/
		if (neighbors[UP] != -1) {
			MPI_Wait(&upRequest[RECV], &upStatus[RECV]);
		}
		if (neighbors[DOWN] != -1) {
			MPI_Wait(&downRequest[RECV], &downStatus[RECV]);
		}
		if (neighbors[LEFT] != -1) {
			MPI_Wait(&leftRequest[RECV], &leftStatus[RECV]);
		}
		if (neighbors[RIGHT] != -1) {
			MPI_Wait(&rightRequest[RECV], &rightStatus[RECV]);
		}
		
		/*compute borders*/
		outerUpdate(chart, size, iz, buffs, neighbors, threadNum);
		
		/*wait until data are transfered before continue with next step*/
		if (neighbors[UP] != -1) {
			MPI_Wait(&upRequest[SEND], &upStatus[SEND]);
		}
		if (neighbors[DOWN] != -1) {
			MPI_Wait(&downRequest[SEND], &downStatus[SEND]);
		}
		if (neighbors[LEFT] != -1) {
			MPI_Wait(&leftRequest[SEND], &leftStatus[SEND]);
		}
		if (neighbors[RIGHT] != -1) {
			MPI_Wait(&rightRequest[SEND], &rightStatus[SEND]);
		}
		
		/*Prepare iz for the next iteration*/
		iz = 1 - iz;
	}
	
}