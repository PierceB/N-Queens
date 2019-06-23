from sys import argv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

markers = ['o', '^', '>', '*']
colours = ['r', 'b', 'g', 'y']

out = open("all.txt", 'w')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

sl_filename = "slurm.8.4.out"
r_filename = "recursive.txt"
total_times = []
solve_times = []
depths = []
depth = 6
sizes = []
times = []

# CUDA RESULTS
for N in range(2, 16):
	filename = "results_{}.txt".format(N)
	f = open(filename, 'r')
	lines = f.readlines()

	if(len(lines) < depth): continue

	data = lines[depth - 1].rstrip("\n").split()
	sizes.append(N)
	times.append(float(data[7]))
	solve_times.append(float(data[9]))

	f.close()

out.write("Total_times")
for i in range(6):
	out.write(":NA")
for i in range(len(times)):
	out.write(":{}".format(times[i]))

out.write("\n")

out.write("Solve_times")
for i in range(6):
	out.write(":NA")
for i in range(len(solve_times)):
	out.write(":{}".format(solve_times[i]))

out.write("\n");

plt.plot(sizes, times, label="CUDA (depth 6) (Total time)", color=colours[0])
plt.plot(sizes, solve_times, label="CUDA (depth 6) (Solve time)", color=colours[3])

#SLURM RESULTS
f = open(sl_filename, 'r')
lines = f.readlines()
sizes = []
times = []
for line in lines:
	data = line.rstrip("\n").split(":")
	sizes.append(int(data[0]))	
	times.append(float(data[1]))

out.write("MPI")
for i in range(len(times)):
	out.write(":{}".format(times[i]))

out.write("\n")

plt.plot(sizes, times, label="MPI (Nodes: 8, Threads per node: 4)", color=colours[1])
f.close()

f = open(r_filename, 'r')
lines = f.readlines()
sizes = []
times = []
for line in lines:
	data = line.rstrip("\n").split()
	sizes.append(int(data[1]))
	times.append(float(data[5]))

f.close()
out.write("Serial")
for i in range(len(times)):
	out.write(":{}".format(times[i]))
out.write("\n")
out.close()

plt.plot(sizes, times, label="Serial (Recursive)", color=colours[2])

plt.legend()
output = "method_comparison_graph".format(N)
plt.title("Board Size vs. Time (Total)".format(N))
plt.xlabel('Board Size')
plt.ylabel('Time (s)')
plt.savefig("{}.png".format(output))
plt.show()
