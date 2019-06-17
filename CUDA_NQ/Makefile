# General NVCC Makefile
CC	= /usr/local/cuda-10.0/bin/nvcc #Points to NVCC on lab computers
LDFLAGS = -L /usr/local/cuda-10.0/lib64 #Included Libraries
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc #Included sample Files

# all: histogram target_name

CUDA_NQ: CUDA_NQ.cu
	$(CC) CUDA_NQ.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) CUDA_NQ.o  $(LDFLAGS) $(IFLAGS) -o CUDA_NQ

# target_name: object_name.cu
	#$(CC) object_name.cu $(LDFLAGS) $(IFLAGS) -c $<
	#$(CC) object_name.o $(LDFLAGS) $(IFLAGS) -o target_name

#clean:
#$(RM) CUDA_NQ *.o *.~CUDA_NQ #add target_name
