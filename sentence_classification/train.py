#! /usr/bin/env python3 

import tensorflow as tf
import numpy as np

import formatting_data

from tensorflow.contrib import learn


# Parameters(set flag. print flag)
# ====================================================================

# Data loading
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
tf.flags.DEFINE_integer("N_fold", 10, "N fold cross validation")
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



# Data formatting(loading, embedding by dictionary, make N fold)
# ====================================================================

# Load data(reading, parsing, labeling data)
print("Loading data...")
x_sentences, y_type = formatting_data.load_data_and_labels(FLAGS.drive_log_file, FLAGS.net_log_file)
# x_sentences = ["drive doesn't working", "network doens't working", ...]
# y_type      = [1, 0],                   [0, 1]                   , ...]
len_y = len(y_type)

# Build vocabulary(embedding)
max_document_length = max([len(sentence.split(" ")) for sentence in x_sentences])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
sentences_indexs = np.array(list(vocab_processor.fit_transform(x_sentences)))
# 'learn.preprocessing.VocablularyProcessor.fit_transform' change word to index
# e.g. When max_document_length == 4
# ["I like pizza", "i don't like pasta", ..] => [[1, 2, 3, 0] [1, 4, 2, 5], ..]

# make N fold(shuffle, split N fold)
for N in range(FLAGS.N_fold):
  x_train, x_val, y_train, y_val = formatting_data.make_N_fold(sentences_indexs, y_type, FLAGS.N_fold)

# Training
# ====================================================================



