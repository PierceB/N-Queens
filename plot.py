from sys import argv

if len(argv) < 2:
	print("Usage: {} [results] [output]".format(argv[0]))	
	exit(1)

import matplotlib.pyplot as plt

filename = argv[1]
output = argv[2]
f = open(filename, 'r')
serial_x = []
serial_souls = []
serial_y = []
f.readline() #Serial
for i in range(15):
	data = f.readline().rstrip("\n").split()
	serial_x.append(i + 1)
	serial_souls.append(int(data[3]))
	serial_y.append(float(data[5]))
plt.plot(serial_x[8:], serial_y[8:], label="Serial")

for i in range(9):
	f.readline()
	y = []
	x = []
	souls = []
	for j in range(15):
		data = f.readline().rstrip("\n").split()
		x.append(j + 1)
		souls.append(int(data[2][:-1]))
		y.append(float(data[3]))
	if i == 0 or i == 1:
		continue
	plt.plot(x[8:], y[8:], label="MPI ({} processes)".format(i + 1))

plt.legend()
plt.title("N (board length) vs. Serial/Parallel times")
plt.xlabel("N (board length)")
plt.ylabel("Time (s)")
#plt.plot(x, y[0], 'ro', x, y[1], 'go')
plt.savefig("{}.png".format(output))
plt.show()
