import numpy as np

class student:
	a = 1
	b = 2


array = [[1, 2, 3], [4, 5, 6]]

print(dir(student))


for i, name in enumerate(['first', 'second', 'third']):
	print (i, name)

array = np.array(array)

print(array[:2])
print(array.shape) # print shape

