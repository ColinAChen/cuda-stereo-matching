C = nvcc
CPP = g++
NVCCFLAGS =  -arch=sm_60 -std=c++11 
CFLAGS = -lX11 -lpthread

all: match

match: match.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o match match.cu 
clean:
	rm -f match match.bmp *.o disparity.bmp

