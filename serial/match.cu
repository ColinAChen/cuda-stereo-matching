#include <stdio.h>
#include <string.h>
#include "CImg.h"
#include <iostream>
#include <chrono>
#include <vector> 

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
float SSD(const int blockSize, const char left[], const char right[]){
	float ssd = 0;
	for (int i = 0; i < blockSize * blockSize; i++){
		ssd += (left[i] - right[i]) * (left[i] -right[i]);
	}
	return ssd;
}

void match(int blockSize, int cols, int rows, const unsigned char left[], const unsigned char right[], int disparity[], char recreate[]){//const unsigned char left[], const unsigned char right[]){
//void match(int blockSize, int width, int height, CImg left, CImg right, CImg disparity){//const unsigned char left[], const unsigned char right[]){
	printf("rows %d, cols %d\n", rows, cols);
	//height: number of rows
	//width: number of columns
	//Implement a basic block matching SSD
	//for each pixel in the left image, find a corresponding pixel in the right image
	//the image are already rectified, so we use the epipolar constraint to narrow down the serach space to corresponding rows
	//we remove some noise by searching a square of pixels instead of individual pixels
	//we match blocks by subtracting one block from the other, squaring each diffference (to normalize), then summing over the block

	//Given a block size N, we need to match width/N blocks if width%N == 0 or width/N + 1 if Nis not 0

	// For each block in the left image, search the corresponding row in the right image for a match
	// Once we find the best match, store the x disparity in the disparity image as a brightness value 
	char leftBlock[blockSize * blockSize];
	char rightBlock[blockSize * blockSize];
	int minDisparity = 0;
	int minSSD = -1;
	int minCol = 0;
	for (int i = 0; i < rows-blockSize; i += blockSize){ //iterate through left row, stride by blockSize
		for (int j = 0; j < cols-blockSize; j+= blockSize){ //iterate through left col, stride by blockSize
			/*if (i > 1990 && j > 2970){
				printf("i: %d, j: %d\n", i, j);
			}*/
			// create a left block for SSD calculation
			for (int row = 0; row < blockSize; row++){
				for (int col = 0; col < blockSize; col++){
					recreate[((i + row) * rows) + j + col] = left[((i + rows) * i) + j + col];

					/*if (i > 1990 && j > 2970){
						printf("left block row: %d, col: %d, index: %d\n", i + row, j + col, ((i + row)* rows) + j + col);
					}*/
					leftBlock[(row * blockSize) + col] = left[((i + row) * rows) + j + col];
					//leftBlock[(row * blockSize) + col] = left(i+row, j + col);
				}
			}
			for (int k = j; k < cols-blockSize; k++){ //search each column in right for a potential match
			// we can start at current row j because it is impossible to find a negative disparity
				//create a right block for SSD
				for (int row = 0; row < blockSize; row++){
					for (int col = 0; col < blockSize; col++){
						if (i > 1990 && j > 2970 && k > 2970){
							//printf("right block row: %d, col: %d, index: %d\n", i + row, j + col, ((i + row)* rows) + j + col + k);
						}
						rightBlock[(row * blockSize) + col] = right[( (i + row) * rows) + col + k];;
						//rightBlock[(row * blockSize) + col] = right(i + row, j + col + k);
					}
				}
				// determine if this block is any good
				float ssd = SSD(blockSize, leftBlock, rightBlock);
				//printf("left row: %d, col: %d, right row: %d, col:%d, ssd: %f\n", i, j, i, j + k, ssd);
				if (minSSD == -1 || ssd < minSSD){
					// This is either the first block we have tested or we found a new lowest SSD
					minSSD = ssd;
					minDisparity = k - j;
					minCol = k;
				}
			}
			//printf("minDisparity: %d, brightness: %d\n", minDisparity, 255 * minDisparity/width);
			// we have searched all possible columns in the right image for a good match
			
			//set the block in the disparity image corresponding to the block in the left image to be scaled based on the determined disparity
			for (int row = 0; row < blockSize; row++){
				for (int col = 0; col < blockSize; col++){
					// brightness = disparity/width * 255. This means high disparity (closer to camera) should be brighter
					/*if (row == 0 && col == 0){
						printf("writing %d at row %d col %d, disparity: %d\n", 255 * minDisparity/width, i + row, col + j,((i + row) * width) + col + j);
					}*/
					/*if (i > 1990 && col > 2970){
						printf("writing %d at row %d col %d, disparity: %d\n", 255 * minDisparity/cols, i + row, col + j,((i + row) * rows) + col + j);
					}*/	
					//disparity[((i + row) * rows) + col + j] = 255 * minDisparity/cols;
					disparity[ ((i + row) * rows) + col + j  ] = right[ ((i + row) * rows) + col + j + minCol];
					//disparity(i + row, j+col) = (minDisparity/width) * 255;
				}
			}
			//disparity[(i * width) + j] = minDisparity;
			// reset the local variables for the next block in the left image
			minSSD = -1;
		}

	}
		
}

void parseCalibration(char* path, float cam0[3][3], float cam1[3][3]){//, float* doffs, float* baseline, int* width, int* height){
	//parse 
	//cam0
	//cam1
	FILE* fp;
	fp = fopen(path, "r");
	int bufferLength = 255;
	char buffer[bufferLength];
	int line = 0;
	
	while(fgets(buffer, bufferLength, fp)){
		if (line == 0 || line == 1){
			// remove [ and ; from matrix
			char* matrix = buffer + 6;
			//printf("matrix: %s\n", matrix);
			char cleanedMatrix[100];
			int i = 0; int j = 0;
			while(matrix[i] != '\0'){
				while (matrix[i] == '[' || matrix[i] == ']' || matrix[i] == ';'){
					i++;	
				}
				cleanedMatrix[j] = matrix[i];
				i++; j++;
			}	
			cleanedMatrix[j] = '\0';
			//printf("cleanedMatrix: %s\n", cleanedMatrix);
			char* token = strtok(cleanedMatrix, " ");
			int row = 0; int col = 0;
			while (token != NULL){
				if (line == 0){
					cam0[row][col] = atof(token);
				}
				if (line == 1){
					cam1[row][col] = atof(token);
				}
				
				token = strtok(NULL, " "); 
				if (col == 2){
					row++;
					col = 0;
				}		
				else{
					col++;
				}			
			}
				
			line++;
		}
	}
	fclose(fp);

	
}
void printMatrix(float matrix[3][3]){
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			printf("%f ", matrix[i][j]);
		}
	printf("\n");
	}

}
int main(){
	/*	
	char* path = "../Bicycle1-perfect/calib.txt";
	char* leftImagePath = "../Bicycle1-perfect/GSim0.png";
	char* rightImagePath = "../Bicycle1-perfect/GSim1.png";
	*/
	//float cam0[3][3]; float cam1[3][3];
	//parseCalibration(path, cam0, cam1);
	//printMatrix(cam0);
	//printMatrix(cam1);

	int blockSize = 64;
	// load image using CImg class constructor
	CImg<unsigned char>leftImage("GSim0.bmp");
	CImg<unsigned char>rightImage("GSim1.bmp");
	int width = leftImage.width();
	int height = leftImage.height();
	printf("images loaded\n");
	CImg<unsigned char>disparity(width, height,1,1);
	CImg<unsigned char>recreate(width, height,1,1);
	printf("disparity image created\n");
	/*
	num cols = width = 32
	num rows = height = 8
	CImg<unsigned char>test(32, 8,1,1);
	printf("set image\n");
	for(int i = 0; i < 8; i++){//columns first
		for(int j = 0; j < 32; j++){// rows
			printf("j: %d, i: %d, brightness: %d\n", j, i, (i * 32) + j);
			test(j,i) = (i * 32) + j;
		}

	}
	printf("read image\n");
	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 32; j++){
			printf("i: %d, j: %d, index: %d, brightness: %d\n", i,j,(i * 32) + j,test.data()[(32 * i) + j]);
		}

	}
	test.save("test.bmp");*/
	//printf("width: %d, height: %d\n", leftImage.width(), leftImage.height());
	int disparities[width * height];
	char reC[width * height];
	printf("starting matching, blockSize: %d\n", blockSize);
	auto start = high_resolution_clock::now();
		//INSERT CODE TO BE TIMED
	match(blockSize,width,height, leftImage.data(), rightImage.data(), disparities, reC);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<nanoseconds>(stop-start);
	cout << duration.count()<< endl;
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
	/*
	char ssd1[] = {1,2,3,4};
	char ssd2[] = {4,3,2,1};
	printf("ssd test: %f\n", SSD(2,ssd1, ssd2));*/	
	return 0;
}



