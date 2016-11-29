import tensorflow as tf
import numpy as np


# 3X4
l = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("origin : \n{}\n".format(l))

l = np.reshape([l], -1)
print("after reshape : \n{}".format(l))


x = list()
x = x + [l[1:3]]
print("x = {}".format(x))



"""
x = 3
for x in range(x):
    print(x)    #print 0~2
"""


