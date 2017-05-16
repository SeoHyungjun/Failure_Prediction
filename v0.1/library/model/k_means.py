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
    def __init__(self, session):
        self.session = session
        self.model_name = ct.KMEANS_MODEL_NAME
        # output config
        self.model_save_tag = ct.MODEL_SAVE_TAG
        self.dir_output = ct.DIR_OUTPUT
        self.trained_centroid_file = ct.KMEANS_TRAINED_CENTROID_FILE
        # create_model
        self.num_centroid = tf.constant(ct.KMEANS_NUM_CENTROID, name="num_centroid").eval()
        self.max_iters = ct.KMEANS_MAX_ITERS
        #x_width
        #x_train
        #x_evaluation
        #x_run

    def set_config(self):
        pass

    def create_model(self, x_width):
        # make output directory
        self.dirpath_trained_model, self.dirpath_summary_train, self.dirpath_summary_validation = set_output_dir.make_dir(self.model_name, self.dir_output)
        self.model_filepath = os.path.join(self.dirpath_trained_model, self.model_save_tag)

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
        self.initial_step = 0
        self.global_step = tf.assign(tf.Variable(0, dtype=tf.int32), self.input_step, name="global_step")

        # Initialize all variables of tensor
        self.session.run(tf.global_variables_initializer())
    
    def restore_all(self):
        dirpath_model = os.path.join(self.dir_output, self.model_name)
        self.dirpath_trained_model = os.path.join(dirpath_model, ct.DIR_TRAINED_MODEL)
        self.dirpath_summary_train = os.path.join(dirpath_model, ct.DIR_SUMMARY, ct.DIR_SUMMARY_TRAIN)
        self.dirpath_summary_validation = os.path.join(dirpath_model, ct.DIR_SUMMARY, ct.DIR_SUMMARY_VALIDATION)
        checkpoint_file_path = os.path.join(self.dirpath_trained_model)
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
        self.initial_step = int(latest_model[-1])
        self.global_step = self.session.graph.get_operation_by_name("global_step").outputs[0]
 
    def train(self, x):
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
        for i in range(self.max_iters):
            feed_dict.update({self.input_step:self.initial_step+i+1})
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
        filepath_trained_model = os.path.join(self.dirpath_trained_model, self.model_save_tag) 
        model_saver.save(self.session, filepath_trained_model, global_step=global_step-1)
        np.savetxt(os.path.join(self.dirpath_summary_train, ct.KMEANS_TRAINED_CENTROID_FILE), updated_centroids, delimiter=',')
        print("Save learned model at step {}".format(global_step-1))
        
    def run(self, x):
        centroids = np.genfromtxt(os.path.join(self.dirpath_summary_train, self.trained_centroid_file), delimiter=',')
        feed_dict = {
                self.input_x : x,
                self.input_centroids : centroids,
        }
        result = self.session.run(self.assignments, feed_dict)
        return result
