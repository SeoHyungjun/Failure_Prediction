import os
import random
import sys
from model import Model

import numpy as np
import tensorflow as tf 
import pandas as pd

import set_output_dir
import constant as ct

class K_Means(Model):
    def __init__(self, session, model_name="k_means", save_tag=ct.STR_SAVED_MODEL_PREFIX):
        # make output directory
        self.model_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(model_name)

        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.model_path, save_tag) 
        self.session = session
  
    def set_config(self):
        pass

    def create_model(self, x_width, num_centroid, max_iters=1000):
        self.num_centroid = tf.constant(num_centroid, name="num_centroid").eval()
        self.input_x = tf.placeholder(tf.float32, [None, x_width], name="input_x")
        self.input_centroids = tf.placeholder(tf.float32, [self.num_centroid, x_width], name="input_centroids")
        self.input_step = tf.placeholder(tf.int32, name="input_step")

        centroids = tf.get_variable("centroids", shape=[self.num_centroid, x_width])
        init_centroids = tf.assign(centroids, self.input_centroids)
        expanded_centroids = tf.expand_dims(init_centroids, 0)
        expanded_point = tf.expand_dims(self.input_x, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_point, expanded_centroids)), -1)
        self.assignments = tf.argmin(distances, -1, name="assignment")
        points_per_centroid = [tf.gather(self.input_x, tf.reshape(tf.where(tf.equal(self.assignments, centroid_index)), [-1])) for centroid_index in range(self.num_centroid)]
        updated_centroids = [tf.reduce_mean(points, reduction_indices=0)
                for points in points_per_centroid]
        self.train_op = tf.assign(centroids, updated_centroids, name="train_op")
        self.sum_distances = tf.reduce_sum(tf.reduce_min(distances, -1), name="sum_distances")
        self.global_step = tf.assign(tf.Variable(0, dtype=tf.int32), self.input_step, name="global_step")

        # Initialize all variables of tensor
        self.session.run(tf.global_variables_initializer())
    
    def restore_all(self, model_name="k_means", dir_root=ct.STR_DERECTORY_ROOT, graph_dir=ct.STR_DERECTORY_GRAPH):
        checkpoint_file_path = os.path.join(dir_root, model_name, graph_dir)
        # Restore graph and variables and operation
        latest_model = tf.train.latest_checkpoint(checkpoint_file_path)
        restorer = tf.train.import_meta_graph("{}.meta".format(latest_model))
        restorer.restore(self.session, "{}".format(latest_model))
        # input operation
        self.num_centroid = self.session.graph.get_operation_by_name("num_centroid").outputs[0].eval()
        self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
        self.input_centroids  = self.session.graph.get_operation_by_name("input_centroids").outputs[0]
        self.input_step  = self.session.graph.get_operation_by_name("input_step").outputs[0]
        # output operation
        self.train_op = self.session.graph.get_operation_by_name("train_op").outputs[0]
        self.sum_distances = self.session.graph.get_operation_by_name("sum_distances").outputs[0]
        self.assignments = self.session.graph.get_operation_by_name("assignment").outputs[0]
        self.global_step = self.session.graph.get_operation_by_name("global_step").outputs[0]
 

    def train(self, x, max_iters, file_output=ct.STR_CENTROID_FILE):
        if self.num_centroid >= len(x):
            print("the number of centroid must be larger than (the number of data + 1)")
            sys.exit()
        # set default centroids randomly
        centroid_indexs = random.sample(range(0, len(x)-1), self.num_centroid)
        centroids_feed = x.iloc[centroid_indexs]
        feed_dict = {
                self.input_x : x,
                self.input_centroids : centroids_feed,
        }
        model_saver = tf.train.Saver(tf.global_variables())

        # start train
        for i in range(max_iters):
            feed_dict.update({self.input_step:i+1})
            updated_centroids, global_step, sum_distances = self.session.run([self.train_op, self.global_step, self.sum_distances], feed_dict)
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
            print("Finish {} st step".format(global_step))
            print("centroid:\n{}".format(updated_centroids))
            print("sum_distances = {}\n".format(sum_distances))
            feed_dict.update({self.input_centroids:updated_centroids})
        print("finish!!\nsum_distances = {}".format(sum_distances))
        model_saver.save(self.session, self.model_prefix, global_step=global_step-1)
        np.savetxt(os.path.join(self.summary_train_path, file_output), updated_centroids, delimiter=',')
        print("Save leanred model at step {}".format(global_step-1))
        
    def run(self, x, file_output=ct.STR_CENTROID_FILE):
        centroids = np.genfromtxt(os.path.join(self.summary_train_path, file_output), delimiter=',')
        feed_dict = {
                self.input_x : x,
                self.input_centroids : centroids,
        }
        result = self.session.run(self.assignments, feed_dict)
        print("cluster result!! = {}".format(result))
