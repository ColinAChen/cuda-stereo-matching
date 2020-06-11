#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector> 

#define BLOCK_SIZE 16

using namespace std::chrono;
using std::cout;
using std::endl;
//Matrix struct from NIVDIA's CUDA programming guide
typedef struct{
	int width;
	int height; 
	float* elements;

}Matrix;

//printer helper function I used to verfiy the output is correct
void printMatrix(float* toPrint,int N){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%5.1f ", toPrint[j + (i * N)]);
		}
		printf("\n");
	}
}
void printVector(float* toPrint, int length){
	for (int i = 0; i < length; i ++){
		printf("%5.1f ", toPrint[i]);
	}
	printf("\n");
}
void matrixVector(float* A, float* B, float* out, int length){
 
	for (int i = 0; i < length; i++){
		float sum = 0;
		for (int j = 0; j < length; j++){
			sum += A[i*length + j] * B[j]; 
		}
		out[i] = sum;
	}
	
}	



int main(){
	//define the size of the matricies to multiply
	int N = 16 * 512; //need to test 16, 128, 1024, 2048, 8192

	//Create a local matrix and vector to load onto the GPU
	//Matrix A;
	//A.width = N; A.height = N;
	//A.elements = (float*)malloc(N*N*sizeof(float));
	float *A;float* B;
	A = (float*)malloc(N* N * sizeof(float));
	B = (float*)malloc(N*sizeof(float));
	// Fill the local matrix and vector with items to multiply
	for (int i = 0; i < N*N; i++){
		if (i < N){
			B[i] = 1.0f;
		}
		A[i] = 1.0f;
	}
	
	// create a vector to hold the output of the matrix-vector multiplication;
	float* out = (float*)malloc(N * sizeof(float));

	auto start = high_resolution_clock::now();
	matrixVector(A, B, out, N);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<nanoseconds>(stop-start);
	cout << duration.count()<< endl;
	//printMatrix(A,N);
	//printVector(B,N);
	//printVector(out,N);
	free(A);
	free(B);
	free(out);
	
	
}
