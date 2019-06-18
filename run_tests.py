from sys import argv
import os

if len(argv) < 3:
	print("Usage: {} [Up to N] [averages] [filename]".format(argv[0]))
	exit(1)

N = int(argv[1])
averages = int(argv[2])
filename = argv[3]
n = 10

#Compile

compile_serial_command = "gcc SERIAL/Serial_NQ.c -o Serial_NQ"
os.system(compile_serial_command)

compile_mpi_command = "mpicc MPI/MPI_NQP.c -o MPI_NQP"
os.system(compile_mpi_command)

command_serial = "./Serial_NQ {} {} {}".format(N, averages, filename)
print("Serial: ")
os.system(command_serial)
print()

for i in range(1, n):
	command_mpi = "mpiexec -n {} ./MPI_NQP".format(i)
	print("MPI ({} procs): ".format(i))
	os.system(command_mpi)
	print()
