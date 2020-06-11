#include <stdio.h>
#include <cuda.h>
#include "CImg.h"


using namespace std;
using namespace cimg_library;


__global__ void blur(const unsigned char d_src[], unsigned char d_dst[], int width, int height){
	// column
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	// row
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
	// make sure we are operating on a pixel within bounds
	if (pos_x == 0 || pos_x >= width-1 || pos_y == 0 || pos_y >= height-1){
		return;
	}
	
	// collect the weighted pixel values and set the output pixel to the sum
	// red color channel
	// from top to bottom, left to right
	d_dst[pos_y * width + pos_x] = (unsigned char)((0.0625f * d_src[(pos_y - 1) * width + (pos_x - 1)]) 
				     + (0.125f * d_src[(pos_y - 1) * width + (pos_x)])
				     + (0.0625f * d_src[(pos_y - 1) * width + (pos_x + 1)])
				     + (0.125f * d_src[(pos_y) * width + (pos_x - 1)])
				     + (0.25f * d_src[(pos_y) * width + pos_x])
				     + (0.125f * d_src[pos_y * width + (pos_x+1)])
				     + (0.0625 * d_src[(pos_y + 1) * width + (pos_x-1)])
				     + (0.125 * d_src[(pos_y + 1) * width + (pos_x)])
				     + (0.0625 * d_src[(pos_y +1) * width + (pos_x + 1)]));
	// blue color channel
	d_dst[(pos_y + height) * width + pos_x] = (unsigned char)((0.0625f * d_src[(pos_y - 1 + height) * width + (pos_x - 1)]) 
				     + (0.125f * d_src[(pos_y - 1 + height) * width + (pos_x)])
				     + (0.0625f * d_src[(pos_y - 1 + height) * width + (pos_x + 1)])
				     + (0.125f * d_src[(pos_y + height) * width + (pos_x - 1)])
				     + (0.25f * d_src[(pos_y + height) * width + pos_x])
				     + (0.125f * d_src[(pos_y + height) * width + (pos_x+1)])
				     + (0.0625 * d_src[(pos_y + 1 + height) * width + (pos_x-1)])
				     + (0.125 * d_src[(pos_y + 1 + height) * width + (pos_x)])
				     + (0.0625 * d_src[(pos_y + 1 + height) * width + (pos_x + 1)]));
	// green color channel
	d_dst[(pos_y + height + height) * width + pos_x] = (unsigned char)((0.0625f * d_src[(pos_y - 1 + height + height) * width + (pos_x - 1)]) 
				     + (0.125f * d_src[(pos_y - 1 + height + height) * width + (pos_x)])
				     + (0.0625f * d_src[(pos_y - 1 + height + height) * width + (pos_x + 1)])
				     + (0.125f * d_src[(pos_y + height + height) * width + (pos_x - 1)])
				     + (0.25f * d_src[(pos_y + height + height) * width + pos_x])
				     + (0.125f * d_src[(pos_y + height + height) * width + (pos_x+1)])
				     + (0.0625 * d_src[(pos_y + 1 + height + height) * width + (pos_x-1)])
				     + (0.125 * d_src[(pos_y + 1 + height + height) * width + (pos_x)])
				     + (0.0625 * d_src[(pos_y + 1 + height + height) * width + (pos_x + 1)]));


}

int main(){
	// load image using CImg class constructor
	CImg<unsigned char>src("ana_sbk.bmp");
	int width = src.width();
	int height = src.height();
	
	size_t size = src.size();
	//size_t blur_size = (width) * (height) * 3 * sizeof(unsigned char);
	
	// create our output image
	// the height and width will be reduced by 2 because of the shape of our stencil
	// width, height, number of channels (3 for color), number of images
	CImg<unsigned char>blur_image(width, height,1,3);		
	// fill the image with black to set the edges to black
	blur_image.fill(0);
	// allocate device memory
	unsigned char *d_src;
	unsigned char *d_blur;
	cudaMalloc((void**)&d_src, size);
	cudaMalloc((void**)&d_blur, size);
	
	//copy the source image to device
	cudaMemcpy(d_src, src.data(), size, cudaMemcpyHostToDevice);
	
	// launch the kernal
	dim3 blkDim (16, 16, 1);
	dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
	blur<<<grdDim, blkDim>>>(d_src, d_blur, width, height);
	
	//copy the blurred image back to host
	cudaMemcpy(blur_image.data(), d_blur, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_src);
	cudaFree(d_blur);
	
	blur_image.save("blur.bmp");
	return 0;
}
