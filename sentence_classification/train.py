#! /usr/bin/env python3 

import tensorflow as tf
import numpy as np

import formatting_data


# Parameters(set flag. print flag)
# ====================================================================

# Data loading
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("drive_log_file", "./data/dict/test_disk", "Data source for the drive log file")
tf.flags.DEFINE_string("net_log_file", "./data/dict/test_net", "Data source for the net log file")

# Model Hyperparameters
tf.flags.DEFINE_integer("word_vector_size", 128, "Vector lenth of each word")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Step size of optimizer")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")

# Tensorflow H/W option
tf.flags.DEFINE_boolean("allow_soft_placemnet", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_plcaement", False, "Log placement of ops on devices")

# print flags
FLAGS = tf.flags.FLAGS          # Make Tensorflow Flag object
FLAGS._parse_flags()            # add flags into "__flags"
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
  print("{}={}".format(attr.upper(), value))
print("")



# Data formatting(parsing, embedding by dictionary, and shuffle)
# ====================================================================

# Load data
print("Loading data...")
x_sentences, y_type = formatting_data.load_data_and_labels(FLAGS.drive_log_file, FLAGS.net_log_file)
# x_sentences = ['drive doesn't working', 'network doens't working', ...]
# y_type      = [1, 0],                   [0, 1]                   , ...]




# Training
# ====================================================================
