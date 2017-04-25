import os
import random
from model import Model

import numpy as np
import tensorflow as tf 
import pandas as pd

import set_output_dir

class K_Means(Model):
    def __init__(self, model_tag, save_tag, session):
        # make output directory
        self.model_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(model_tag)

        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.model_path, save_tag) 
        self.session = session
  
    def create_model(self, x_width, num_centroid, MAX_ITERS=1000):
        self.num_centroid = num_centroid
        self.input_x = tf.placeholder(tf.float32, [None, x_width], name="input_x")
        self.input_centroid = tf.placeholder(tf.float32, [self.num_centroid, x_width], name="input_centroid")

        centroid = tf.get_variable("centroid", shape=[self.num_centroid, x_width])
        update = tf.assign(centroid, self.input_centroid)
        expanded_centroid = tf.expand_dims(update, 1)
        expanded_point = tf.expand_dims(self.input_x, 0)
        test = tf.subtract(expanded_point, expanded_centroid)
        # for pre_cent != cent
            # set centroid point
            # map each point to nearest centroid
        # Initialize all variables of tensor
        self.session.run(tf.global_variables_initializer())

        x = pd.read_csv("./input.csv")
        center_index = random.sample(range(0, len(x)-1), self.num_centroid) 
        print("###center_index is {}".format(center_index))
        feed_centroid = x.iloc[center_index]
        print("###center is {}".format(feed_centroid))
        feed_dict = {
                self.input_x : x,
                self.input_centroid : feed_centroid
        }
        output = self.session.run([test], feed_dict)
        print(output)
    
    def restore_all(self):
        pass
  
    def train(self, x):
        # select initial centroid
        """
        centroid = x.iloc[random.sample(range(0, len(x)-1), self.num_centroid)]
        feed_dict = {
                self.input_x : x,
                self.input_centroid : centroid
        }
        print(self.session.run([self.expanded_point], feed_dict))
        """
    def run(self):
        pass

