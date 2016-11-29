import tensorflow as tf
import numpy as np

x = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(4))

print("shuffle_indices = {}".format(shuffle_indices))
