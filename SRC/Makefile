# General NVCC Makefile
CC	= /usr/local/cuda-10.0/bin/nvcc #Points to NVCC on lab computers
LDFLAGS = -L /usr/local/cuda-10.0/lib64 #Included Libraries
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc #Included sample Files

all: SERIAL_RE SERIAL_IT CUDA_NQ MPI_NQP 

SERIAL_RE: SERIAL_RE.c
	gcc SERIAL_RE.c -o SERIAL_RE.out

SERIAL_IT: SERIAL_IT.c
	gcc SERIAL_IT.c -o SERIAL_IT.out

CUDA_NQ: CUDA_NQ.cu
	$(CC) CUDA_NQ.cu  $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) CUDA_NQ.o   $(LDFLAGS) $(IFLAGS) -o CUDA_NQ.out

MPI_NQP: MPI_NQP.c
	mpicc MPI_NQP.c -o MPI_NQP.out

# target_name: object_name.cu
	#$(CC) object_name.cu $(LDFLAGS) $(IFLAGS) -c $<
	#$(CC) object_name.o $(LDFLAGS) $(IFLAGS) -o target_name

clean:
	$(RM) *.o *.out #add target_name
