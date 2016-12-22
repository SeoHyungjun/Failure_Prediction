#import matplotlib.pyplot as plt % matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf

#import seaborn as sns

num_vectors = 1000
num_clusters = 3
num_steps = 100
vector_values = []

for i in range(num_vectors):
	if np.random.random() > 0.5:
		vector_values.append([np.random.normal(0.5, 0.6), np.random.normal(0.3, 0.9),
							np.random.normal(0.7, 0.3), np.random.normal(0.1, 0.4), np.random.normal(1.4, 0.4)])
	else:
		vector_values.append([np.random.normal(2.5, 0.4), np.random.normal(0.8, 0.5),
							np.random.normal(1.4, 0.4), np.random.normal(1.8, 0.2), np.random.normal(2.1, 0.1)])

print(np.shape(vector_values))

fp = open("samples", "w")
for i in vector_values :
#print(round(i[0],5), round(i[1],5), round(i[2],5), round(i[3],5), round(i[4]),5)
	fp.write(str(round(i[0],5)) + ","  +
			str(round(i[1],5)) + "," +
			str(round(i[2],5)) + "," +
			str(round(i[3],5)) + "," +
			str(round(i[4],5)) + "\n")
fp.close()

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

fc0 = open("c0_data", "w")
fc1 = open("c1_data", "w")
fc2 = open("c2_data", "w")

for i in range(len(assignment_values)):
		clustered_data["d0"].append(vector_values[i][0])
		clustered_data["d1"].append(vector_values[i][1])
		clustered_data["d2"].append(vector_values[i][2])
		clustered_data["d3"].append(vector_values[i][3])
		clustered_data["d4"].append(vector_values[i][4])
		clustered_data["cluster"].append(assignment_values[i])

		if assignment_values[i] == 0 :
			fc0.write(str(round(vector_values[i][0],5)) + ","  +
					str(round(vector_values[i][1],5)) + ","  +
					str(round(vector_values[i][2],5)) + ","  +
					str(round(vector_values[i][3],5)) + ","  +
					str(round(vector_values[i][4],5)) + "\n")
		elif assignment_values[i] == 1 :
			fc1.write(str(round(vector_values[i][0],5)) + ","  +
					str(round(vector_values[i][1],5)) + ","  +
					str(round(vector_values[i][2],5)) + ","  +
					str(round(vector_values[i][3],5)) + ","  +
					str(round(vector_values[i][4],5)) + "\n")
		else :
			fc2.write(str(round(vector_values[i][0],5)) + ","  +
					str(round(vector_values[i][1],5)) + ","  +
					str(round(vector_values[i][2],5)) + ","  +
					str(round(vector_values[i][3],5)) + ","  +
					str(round(vector_values[i][4],5)) + "\n")




print("Centriods")
print(centroid_values)


