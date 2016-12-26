#import matplotlib.pyplot as plt % matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

#import seaborn as sns

if len(sys.argv) != 2 :
	print("Input the cluster num.... exit...")
	exit(1)

num_clusters = int(sys.argv[1])
num_steps = 100
vector_values = []

data = pd.read_csv("5attr_normalized.csv", parse_dates=['date'], index_col='date')
data = data.dropna()
vector_values = data.values.tolist()

vectors = tf.constant(vector_values)

centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [num_clusters,-1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)


means = tf.concat(0, [
			tf.reduce_mean(
				tf.gather(vectors,
							tf.reshape(
								tf.where(
									tf.equal(assignments, c)
								), [1, -1])
						), reduction_indices=[1])
			for c in range(num_clusters)])

update_centroids = tf.assign(centroids, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(num_steps) :
	_, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

clustered_data = {"d0" : [], "d1" : [], "d2" : [], "d3" : [], "d4" : [], "cluster" : []}

fp_clusters = []
file_name = "cluster_"

for i in range(0, num_clusters) :
	fname = file_name + str(i)
	fp_clusters.append(open(fname, "w"))

for i in range(len(assignment_values)):
	clustered_data["d0"].append(vector_values[i][0])
	clustered_data["d1"].append(vector_values[i][1])
	clustered_data["d2"].append(vector_values[i][2])
	clustered_data["d3"].append(vector_values[i][3])
	clustered_data["d4"].append(vector_values[i][4])
	clustered_data["cluster"].append(assignment_values[i])

	for j in range(0, num_clusters) :
		if assignment_values[i] == j :
			fp_clusters[j].write(str(vector_values[i][0]) + ","  +
					str(vector_values[i][1]) + ","  +
					str(vector_values[i][2]) + ","  +
					str(vector_values[i][3]) + ","  +
					str(vector_values[i][4]) + "\n")

print("Centriods")
print(centroid_values)

