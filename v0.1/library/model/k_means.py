import os
import random
import sys
from model import Model

import numpy as np
import tensorflow as tf 
import pandas as pd

import set_output_dir

class K_Means(Model):
    def __init__(self, model_name, save_tag, session):
        # make output directory
        self.model_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(model_name)

        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.model_path, save_tag) 
        self.session = session
  
    def set_config(self):
        pass

    def create_model(self, x_width, num_centroid, MAX_ITERS=1000):
        self.num_centroid = num_centroid
        self.input_x = tf.placeholder(tf.float32, [None, x_width], name="input_x")
        self.input_centroids = tf.placeholder(tf.float32, [self.num_centroid, x_width], name="input_centroids")

        centroids = tf.get_variable("centroids", shape=[self.num_centroid, x_width])
        init_centroids = tf.assign(centroids, self.input_centroids)
        expanded_centroids = tf.expand_dims(init_centroids, 0)
        expanded_point = tf.expand_dims(self.input_x, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_point, expanded_centroids)), -1)
        assignments = tf.argmin(distances, -1)
        self.train_op = [tf.reduce_mean(tf.gather(self.input_x, tf.reshape(tf.where(tf.equal(assignments, centroid_index)), [-1])), reduction_indices=0, name="train_op") 
                for centroid_index in range(self.num_centroid)]
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # Initialize all variables of tensor
        self.session.run(tf.global_variables_initializer())
    
    def restore_all(self, model_name):
        checkpoint_file_path = os.path.join(ct.STR_DERECTORY_ROOT, model_name, ct.STR_DERECTORY_GRAPH)
        # Restore graph and variables
        latest_model = tf.train.latest_checkpoint(checkpoint_file_path)
        restorer = tf.train.import_meta_graph("{}.meta".format(latest_model))
        restorer.restore(self.session, "{}".format(latest_model))
        # Restore input operation
        self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
        self.input_centroids  = self.session.graph.get_operation_by_name("input_centroids").outputs[0]
        # Restore train operation
        self.train_op = self.session.graph.get_operation_by_name("train_op").outputs[0]
        self.global_step = self.session.graph.get_operation_by_name("global_step").outputs[0]
  
    def train(self, x, max_iters):
        if self.num_centroid >= len(x):
            print("the number of centroid must be larger than (the number of data + 1)")
            sys.exit()
        centroid_indexs = random.sample(range(0, len(x)-1), self.num_centroid) 
        centroids_feed = x.iloc[centroid_indexs]
        feed_dict = {
                self.input_x : x,
                self.input_centroids : centroids_feed
        }
        # start train
        for i in range(max_iters):
            print("{}st step".format(i))
            updated_centroids = self.session.run([self.train_op], feed_dict)[0]
            # check centroids are changed
            flag_compare_center = []
            if i != 0:
                for j, centroid in enumerate(updated_centroids):
                    if all((feed_dict[self.input_centroids][j] == centroid[j])):
                        flag_compare_center.append(True)
                    else:
                        flag_compare_center.append(False)
                        break
                # if not changed, stop training
                if all(flag_compare_center):
                    break
            feed_dict.update({self.input_centroids:updated_centroids})
        print(updated_centroids)
        print(feed_dict[self.input_centroids])

    def run(self):
        pass

