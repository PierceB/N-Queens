from sys import argv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

markers = ['o', '^', '>', '*']
colours = ['r', 'b', 'g', 'y']

for N in [2*i for i in range(1,5)]:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	souls = []
	marker = markers[int(N/2 - 1)]
	for n in [2*j for j in range(1,5)]:
		Ns = []
		ns = []
		bn = []
		times = []

		colour = colours[int(n/2 - 1)]
		filename = "slurm.{}.{}.out".format(N, n)
		f = open(filename, 'r')
		for it in range(15):
			data = f.readline().rstrip("\n").split(":")
			Ns.append(N)
			ns.append(n)
			bn.append(int(data[0]))
			try:
				t = float(data[1])
			except:
				t = 0.0
			times.append(t)
			souls.append(int(data[2]))

		f.close()
		ax.plot(bn, ns, times, colour)

	output = "nodes_{}".format(N)
	ax.set_title("(Nodes: {}) Board length and Threads vs. MPI Cluster times".format(N))
	ax.set_xlabel('Board length')
	ax.set_ylabel('Threads')
	ax.set_zlabel('Time (s)')

	plt.savefig("{}.png".format(output))
	plt.show()
