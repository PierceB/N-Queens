import matplotlib.pyplot as plt

f = open("p.txt", 'r')
n = int(f.readline().rstrip("\n"))
x = []
y = [[], []]
for i in range(n):
	data = f.readline().rstrip("\n").split()
	x.append(int(data[0]))
	y[0].append(round(float(data[1]), 2))
	y[1].append(round(float(data[2]), 2))

plt.title("Dimension vs. Serial/Parallel times")
plt.xlabel("Dimension")
plt.ylabel("Time (ms)")
plt.plot(x, y[0], 'ro', x, y[1], 'go')
plt.savefig("fig.png")
plt.show()
