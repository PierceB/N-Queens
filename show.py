

def for_loop(i, depth):
	if i == depth:
		return
	print(i)
	for_loop(i + 1, depth)

for_loop(0, 5)
