C = nvcc
CPP = g++
NVCCFLAGS =  -arch=sm_60 -std=c++11 
CFLAGS = -lX11 -lpthread

all: match

match: match.cu bmp.o
	$(C) $(NVCCFLAGS) $(CFLAGS) -o match match.cu bmp.o
bmp.o: bmp.c bmp.h
	$(C) $(CFLAGS) -c bmp.c

clean:
	rm -f match match.bmp *.o disparity.bmp out 

