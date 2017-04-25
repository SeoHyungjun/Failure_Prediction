import os
import random

import numpy as np
import tensorflow as tf 
from model import Model

import set_output_dir

class K_Means(Model):
    def __init__(self, model_tag, save_tag, session):
        # make output directory
        self.model_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(model_tag)

        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.model_path, save_tag) 
        self.session = session
  
    def create_model(self, num_centroid, x_width, MAX_ITERS=1000):
        self.input_x = tf.placeholder(tf.float32, [None, x_width], name="input_x")
        self.input_centroid = tf.placeholder(tf.float32, [num_centroid, x_width], name="input_centroid")
        centroid = tf.Variable(self.input_centroid)

        # for pre_cent != cent
            # set centroid point
            # map each point to nearest centroid
    
    def restore_all(self):
        pass
  
    def train(self, x):
#        print(random.sample(range(0, len(x)-1), num_centroid))
 #       print(x.iloc[3])
        feed_dict = {
                self.input_x : x
#                self.input_centroid : 
        }
        # select initial centroid
  
    def run(self):
        pass



