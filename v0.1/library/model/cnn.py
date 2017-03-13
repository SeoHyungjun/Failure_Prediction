from model import *
import tensorflow as tf
import numpy as np

class CNN(Model):
    ### CV parameter ###

    def __init__(self):
    ### env parameter(init) ###
        pass


    def create_model(self, input_size, num_NN_nodes, num_output, filter_sizes, num_filters, dropout_keep_prob=1.0, l2_reg_lambda=0.0):
    # Model parameter of create_model
    # input_size : size of input matrix(two-dimention), [height, width]  e.g. [3,4]
    # num_NN_nodes : fully connected NN nodes(array)  e.g. [3,4,5,2]
    # num_output : the number of output nodes. if regression, num_output = 1.
    # filter_sizes : list of size of filter matrix(two-dimention)  e.g. [[1,2], [2,3], ...]
    # numb_filters : the number of each size of filter
    # regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1]], name="input_x")
        self.expanded_input_x = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(tf.float32, [None, num_output], name="input_y" )
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
       
        # Keeping track of 12 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        pooled_outputs = []
    
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size[0])):
                # Convolution Later
                filter_shape = [filter_size[0], filter_size[1], 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        
                conv = tf.nn.conv2d(
                    self.expanded_input_x,
                    W,                  # filter
                    strides=[1,1,1,1],
                    padding="VALID",    # no padding
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, input_size[0] - filter_size[0] + 1, input_size[1] - filter_size[1] + 1, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
        
        self.aaa = pooled_outputs
    

    def restore(self):
        pass
    
    def save(self):
        print ("Save model trained!!!")
        pass
      
    def eval(self):
        print ("Eval model trained!!!")
        pass


    def train(self, dev_sample_percentage, data_file_location, out_subdir, tag, batch_size, num_epochs, evaluate_every, checkpoint_every):
    ### train parameter ###
    # dev_sample_percentage : percentage of the training data to use for validation"
    # data_file_location : Data source for training
    # out_subdir : directory for saving output
    # tag : added in output directory name
    # batch_size : Batch Size
    # num_epochs : Number of training epochs
    # evaluate_every : Evaluate model on dev set after this many sters (default: 150)
    # checkpoint_every : Save model after this many steps (default: 150)
    # allow_soft_placement : Allow device soft device placement
    # log_device_placement : Log placement of ops on devices
    # =============================== ###  
        pass
  
  
    def run(self):
        print ("Predict Something!!!!")
        pass

