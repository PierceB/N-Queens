import os
from sys import argv

N = int(argv[1])
filename = "results_{}.txt".format(N)
os.system("make")

for i in range(1, N):
	command = "./CUDA_NQ {} >> {}".format(i, filename)
	os.system(command)
