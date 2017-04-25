import numpy as np
import os
import tensorflow as tf

from model import Model
import constant as ct
import set_output_dir

class K_Means(Model):
    def __init__(self, algorithm_name, session):
        # make output directory
        self.model_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(algorithm_name)
        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.model_path, ct.STR_SAVED_MODEL_PREFIX) 
        self.session = session
        pass
  
    def create_model(self):
        pass
    
    def restore_all(self):
        pass
  
    def train(self):
        pass
  
    def run(self):
        pass



