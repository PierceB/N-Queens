from sys import argv

if len(argv) < 1:
	print("Usage: {} [output]".format(argv[0]))	
	exit(1)

import matplotlib.pyplot as plt

filename_r = "recursive.txt"
filename_i = "Iterative_serial_results.txt"
output = argv[1]
r_x = []
r_souls = []
r_times = []

i_x = []
i_souls = []
i_times = []

f = open(filename_r, 'r')
f.readline() #Recursive
for i in range(15):
	data = f.readline().rstrip("\n").split()
	r_x.append(int(data[1]))
	r_souls.append(int(data[3]))
	r_times.append(float(data[5]))

f.close()
f = open(filename_i, 'r')
f.readline() #Iterative
for i in range(15):
	data = f.readline().rstrip("\n").split()
	i_x.append(int(data[1]))
	i_souls.append(int(data[3]))
	i_times.append(float(data[5]))

f.close()

plt.plot(r_x, r_times, label="Recursive")
plt.plot(i_x, i_times, label="Iterative")

plt.legend()
plt.title("N (board length) vs. Iterative/Recursive times")
plt.xlabel("N (board length)")
plt.ylabel("Time (s)")
#plt.plot(x, y[0], 'ro', x, y[1], 'go')
plt.savefig("{}.png".format(output))
plt.show()
