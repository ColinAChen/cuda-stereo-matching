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


__global__ void minSSD (const float * d_in, float * d_out, int* index_out)
{
	// sdata is allocated in the kernel call : via dynamic shared memeory
	extern __shared__ float sdata [];
	int myId = threadIdx.x + blockDim.x* blockIdx.x;
	int tid = threadIdx.x;
	// load shared mem from global mem
	sdata [tid ] = d_in [ myId ];
	__syncthreads () ; // always sync before using sdata
	// do reduction over shared memory
	int ib = blockDim.x/2;
	while (ib != 0) {
		if(tid < ib && sdata[tid + ib] < sdata[tid])
			sdata[tid] = sdata[tid + ib]; 
			index_out[tid] = tid + ib;

		__syncthreads();

		ib /= 2;
	}

	if(tid == 0)
		d_out[blockIdx.x] = sdata[0];
		index_out[blockIdx.x] = threadIdx.x;
	
}


// blockSize: width of the matching block for SSD
// cols: number of columns in the image
// rows: number of rows in the image
// left[]: array of pixel values in the left image
// right[]: array of pixel values in the right image
// disparity[]: array of pixel values that will hold the disparity values based on minumum SSD
__global__ void match(const int blockSize, const int cols, const int rows, const unsigned char left[], const unsigned char right[], unsigned char disparity[]){
	// we can reduce this algorithm by
	// for each block in the left image:
	// each thread writes the SSD of the left block and their right block
	// reduce on the list of SSDs to find the minimum
	
	// column
	int thread_col = blockIdx.x * blockDim.x + threadIdx.x;
	// row
	int thread_row = blockIdx.y * blockDim.y + threadIdx.y;
	// make sure we are operating on a pixel within bounds
	if (thread_col == 0 || thread_col >= cols-1 || thread_row == 0 || thread_row >= rows-1){
		return;
	}
	
	for (int i = 0; i < rows; i += blockSize){
		for (int j = 0; j < cols; j += blockSize){
			//each thread loads the block from the left image corresponding to loop
			char leftBlock[blockSize * blockSize];	
			for (int row = 0; row < blockSize; row++){
				for (int col = 0; col < blockSize; col++){
					leftBlock[(row * blockSize) + col] = left[(row * rows) + j + col];
				}
			}
			//each thread loads the block from the right image corresponding to its id
			//only need load blocks from the right image that are at least j col across because we cannot find negative disparity

			if (thread_col >= j){
				char rightBlock[blockSize * blockSize];
				for (int row = 0; row < blockSize; row++){
					for (int col = 0; col < blockSize; col++){
						rightBlock[(row * blockSize) + col] = right[((row + thread_row) * rows) + j + col + thread_col];
					}
				}
			}
				
			//each thread computes the SSD and writes it to a shared memory bank
			extern __shared__ float sharedSSD[];
			extern __shared__ float SSDOut[];
			extern __shared__ int index_out[]; //store the index of the minimum
			sharedSSD[thread_row * blockDim.x + thread_col] = SSD(leftBlock,rightBlock);
			//syncthreads to ensure all threads are done finding their SSD
			__syncthreads();
			//call the reduce min kernal to find the minimum SSD and corresponding disparity for this block in the left
			minSSD<<<cols-j,1>>>(sharedSSD, SSDOut, index_out);
			//write the corresponding disparity value for all the pixels in the block in the disparity image
			for (int row = 0; row < blockSize; row++){
				for (int col = 0; col < blockSize; col++){
					disparity[(row * blockSize) + col] = index_out[0];
				}
			}		
		}
	}
}
int main(){

	int blockSize = 64;
	
	// load image using CImg class constructor
	CImg<unsigned char>leftImage("GSim0.bmp");
	CImg<unsigned char>rightImage("GSim1.bmp");
	int width = leftImage.width();
	int height = leftImage.height();
	int size = leftImage.size();
	printf("images loaded\n");
	
	// create the image to hold disparity values
	CImg<unsigned char>disparity(width, height,1,1);
	//CImg<unsigned char>recreate(width, height,1,1);
	printf("disparity image created\n");
	//int disparities[width * height];
	//char reC[width * height];
	printf("starting matching, blockSize: %d\n", blockSize);
	
	
	
	// allocate device memory
	unsigned char *device_left;
	unsigned char *device_right;
	unsigned char *device_disparity;
	cudaMalloc((void**)&device_left, size);
	cudaMalloc((void**)&device_right, size);
	cudaMalloc((void**)&device_disparity, size);
	//copy the left, right, and disparity images to the device
	cudaMemcpy(device_left, leftImage.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_right, rightImage.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_disparity, disparity.data(), size, cudaMemcpyHostToDevice);
	


	// launch the kernal
	dim3 blkDim (16, 16, 1);
	dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//CODE TO TIME
	
	match<<<grdDim, blkDim>>>(blockSize, width, height, leftImage.data(), rightImage.data(), disparity.data());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("test size: %5d     time elapsed: %f\n", width, milliseconds);
		
	//copy the disparity image back to host
	cudaMemcpy(disparity.data(), device_disparity, size, cudaMemcpyDeviceToHost);
	
	cudaFree(device_left);
	cudaFree(device_right);
	cudaFree(device_disparity);
	
	disparity.save("disparity.bmp");
	return 0;

	//used to test the reduce min kernal
	/*
	float* host_A;
	float* host_B;
	float* device_A;
	float* device_B;
	
	int testSize = 256;
	int threadsPerBlock = 64;
	int blocksPerGrid = 64;
	

	printf("testSize: %d\n", testSize);	
	dim3 block_dim = (64,1,1);
	dim3 grid_dim = (64,1,1);
	
	int size = testSize * sizeof(float);
	int sb = blocksPerGrid * sizeof(float);
	
	host_A = (float*)malloc(size);
	host_B = (float*)malloc(sb);
	//fill initial values of A
	for (int i = 0; i < testSize; i++){
		host_A[i] = testSize;
		printf("setting: %d, to %f\n",i,testSize);	
	}
	cudaMalloc((void**)&device_A, size);
	cudaMalloc((void**)&device_B, sb);
	
	cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
	//cudaMalloc((int**)&testMinIndex, 1000 * sizeof(int));
	int sm = threadsPerBlock * sizeof(float);
	minSSD<<<block_dim, grid_dim>>>(device_A, device_B, testSize);
	
	cudaMemcpy(host_B, device_B, sb, cudaMemcpyDeviceToHost);
	for (int i = 0; i < testSize; i++){	

		printf("%f\n",host_B[i]);
	}
	cudaFree(device_A);
	cudaFree(device_B);
	free(host_A);
	free(host_B);
	*/	



}



