#include <stdio.h>
#include <string.h>
#include "CImg.h"
#include <iostream>
#include <chrono>
#include <vector> 
#include <cuda.h>

extern "C" {
  #include "bmp.h"
}
#define BLOCK_SIZE 16

using namespace std::chrono;
using std::cout;
using std::endl;
using namespace std;
using namespace cimg_library;

// Calculate the Sum of Square Differences (SSD) between two blocks
// ARGUMENTS:
// blockSize (int): the size of the matching blocks
// left: an array of pixel values from the left image
// right: an array of pixel values from the right image
// RETURNS:
// ssd (int): the sum of square differences between left and right
__device__ float SSD(const int blockSize, const char left[], const char right[]){
	float ssd = 0;
	for (int i = 0; i < blockSize * blockSize; i++){
		ssd += (left[i] - right[i]) * (left[i] -right[i]);
	}
	return ssd;
}
// blockSize: width of the matching block for SSD
// cols: number of columns in the image
// rows: number of rows in the image
// left[]: array of pixel values in the left image
// right[]: array of pixel values in the right image
// disparity[]: array of pixel values that will hold the disparity values based on minumum SSD
__global__ void match(int blockSize, int cols, int rows, const unsigned char left[], const unsigned char right[], char disparity[]){
	// we can reduce this algorithm by
	// for each block in the left image:
	// each thread writes the SSD of the left block and their right block
	// reduce on the list of SSDs to find the minimum
	
	for (int i = 0; i < rows; i += blockSize){
		for (int j = 0; j < cols; j += blockSize){
			
		}
	}
}

/*// use a reduce function to find the minimum SSD
__global__ void minSSD(const float[] ssd, float* min){
	int myId = thread.x + blockDim.x * blockDim.x;
	int tid = threadIDx.x;
	
	//reduction over global memory
	for (unsigned int s = blockDim.x/2; s > 0; s >>
igned int s = blockDim .x /2; s > 0; s > >= 1)
{
if(tid < s)
{
d_in [ myId ] += d_in [ myId + s];
}
__syncthreads () ; // Maske sure all adds at one stage
}
// only thread 0 writes result for this block back to global mem
if( tid == 0)
{
d_out [ blockIdx .x] = d_in [ myId ];
}i
}
*/
// shared memory reduce to find minimum
__global__ void minSSD ( const float * d_in, float * d_out , int * i_out )
{
	// sdata is allocated in the kernel call : via dynamic shared memeory
	extern __shared__ float sdata [];
	int myId = threadIdx .x + blockDim .x* blockIdx.x;
	int tid = threadIdx .x;
	// load shared mem from global mem
	sdata [tid ] = d_in [ myId ];
	__syncthreads () ; // always sync before using sdata
	// do reduction over shared memory
	for ( int s = blockDim.x/2; s >0; s >>=1)
	{
		if(tid < s)
		{
			sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
			printf("sdata at tid: %d is %f\n", sdata[tid], tid);
			//sdata [tid ] += sdata [ tid + s];
		}
		__syncthreads() ; // make sure all additions are finished
	}
	// only tid 0 writes out result !
	if( tid == 0)
	{
		d_out [ blockIdx .x] = sdata [0];
		i_out[blockIdx.x] = blockIdx.x;
	}
}

int main(){

	int blockSize = 64;
	// load image using CImg class constructor
	CImg<unsigned char>leftImage("GSim0.bmp");
	CImg<unsigned char>rightImage("GSim1.bmp");
	int width = leftImage.width();
	int height = leftImage.height();
	printf("images loaded\n");
	CImg<unsigned char>disparity(width, height,1,1);
	//CImg<unsigned char>recreate(width, height,1,1);
	printf("disparity image created\n");
	int disparities[width * height];
	char reC[width * height];
	printf("starting matching, blockSize: %d\n", blockSize);
	float testMin[1000];
	float* testMinVal; int* testMinIndex;
	for (int i = 0; i < 1000; i++){
		testMin[i] = 1000-i;
	}	

	dim3 block_dim = (1000,1,1);
	dim3 grid_dim = (1,1,1);
	minSSD<<< block_dim, grid_dim>>>(testMin, testMinVal, testMinIndex);
	printf("minVal: %f, at index: %d\n", *testMinVal, *testMinIndex);
	/*
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//CODE TO TIME
	// Perform SAXPY on 1M elements
	//saxpy<<<(N +255)/256 , 256 >>>(N, 2.0f, d_x , d_y);
	//render<<< grid_Dim, block_Dim >>>(image, width, height, max_iter);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("test size: %5d     time elapsed: %f\n", width, milliseconds);


	printf("matching complete\n");
	// fill the disparity image with the found disparity values
	
	for (int i = 0; i < height; i++){// number of rows
		
		for (int j = 0; j < width; j++){ //number of columns
			disparity(j,i) = disparities[(i * width) + j];
			recreate(j,i) = reC[(i * width) + j];
		}
	}
	//Save the disparity image
	disparity.save("disparity.bmp");
	recreate.save("recreate.bmp");
	printf("file saved\n");
	*/
	return 0;
}



