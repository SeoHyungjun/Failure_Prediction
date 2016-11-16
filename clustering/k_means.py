#### K-means ####
# 1. set K(initial center)
# 2. allocate every data to K
# 3. update : calculate new K
# ==============================

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf

#graph package
import seaborn as sns


num_vectors = 1000
num_clusters = 3
num_steps = 100
vector_values = []

#### data generate ####
# xrange(written in book) don't work
# np.random.random() : return random float in the half-open interval [0.0, 1.0)
for i in range(num_vectors):
  if np.random.random() > 0.5:
    vector_values.append([np.random.normal(0.6, 0.1),
                          np.floor(np.random.normal(0.5, 0.3))])
  else:
    vector_values.append([np.random.normal(2.5, 0.4),
                         np.floor(np.random.normal(1.3, 0.5))])

df = pd.DataFrame({"x": [v[0] for v in vector_values], "y": [v[1] for v in vector_values]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=7)
plt.show()

#moving every data to tensor
vectors = tf.constant(vector_values)

##### 1. set K #####
# tf.Variable : create variable
# tf.random_shuffle : shuffle element based on the first dimension
# select random center as many as "num_clusters"
#'input' is  [  [[1, 1, 1], [2, 2, 2]], 
#		[[3, 3, 3], [4, 4, 4]], 
#		[[5, 5, 5], [6, 6, 6]]  ]
#tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
#tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
#                                            [4, 4, 4]]]
#tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
#                                           [[5, 5, 5]]]
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [num_clusters,-1]))
##### 1. set K #####


##### 2. allocate K #####
# add dimension
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)
#print(expanded_vectors.get_shape())    (1000, 2) -> (1, 1000, 2)
#print(expanded_centroids.get_shape())     (3, 2) -> (3, 1, 2)

#tf.sub : [ [center1-data1], [center1-data2], [C1-D3], .... [C1-D1000],
#	    [C2-D1], [C2-D2], ... [C2-D1000],
#	    [C3-D1], [C3-D2], ... [C3-D1000]  ]
# Dimension= (3,1000,2)
#tf.square : ^2
#tf.reduce_sum (ex) (200,300) => (500), (100,200,300) => (600)
#tf.reduce_sum : distance [ C1~D1, C1~D2, ....
#                           C2~D1, C2~D2, ....
#                           C3~D1, C3~D2, .... ]  Dimenstion = (3, 1000)
#tf.argmin : return index which have minimum value
# Dimension= (1000), [0, 1, 2, 2, 0, 1, 2 .....]         
distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)
##### 2. allocate K #####


##### 3. new K #####
#tf.equal : return true where equal
#tf.where : return index where the element is true
#tf.reshape : center1 [3, 55, 100, ....], center2 [1, 20, ...]
#tf.gather(index -> real) : gather data from index(tf.reshape)
#  [3, 55, 100, ....]  -> [ [200,100], [100, 200], ....]
#tf.reduce_mean : calculate means in serveral center
#tf.concat : concat new center [[100,100] , [200,200], [300,300]]
means = tf.concat(0, [
  tf.reduce_mean(
      tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),
		reduction_indices=[1])
  for c in range(num_clusters)])

update_centroids = tf.assign(centroids, means)
##### 3. new K #####

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(num_steps):
  _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

print("centroids")
print(centroid_values)

data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
    data["x"].append(vector_values[i][0])
    data["y"].append(vector_values[i][1])
    data["cluster"].append(assignment_values[i])

for i in range(0, 3) :
	data["x"].append(centroid_values[i][0])
	data["y"].append(centroid_values[i][1])
	data["cluster"].append(3)

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df,fit_reg=False, size=7, hue="cluster", legend=False)
plt.show()
