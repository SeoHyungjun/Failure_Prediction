import numpy as np

batches = list(zip([[1,2,3],[4,5,6],[7,8,9],[7,7,7]], [11,12,13,14]))

#for batch in batches:
x_batch, y_batch = zip(*batches)
print("batches = {}".format(batches))
print("x_batch = {}, y_batch = {}".format(x_batch,y_batch))
