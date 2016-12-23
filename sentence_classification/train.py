#! /usr/bin/env python3 

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import sys

import formatting_data
from model_CNN import TextCNN


# Parameters(set flag. print flag)
# ====================================================================

# Data loading
tf.flags.DEFINE_string("drive_log_file", "./data/dict/test_disk", "Data source for the drive log file")
tf.flags.DEFINE_string("net_log_file", "./data/dict/test_net", "Data source for the net log file")

# Model Hyperparameters
tf.flags.DEFINE_integer("word_vector_size", 128, "lenth of each word vector")
tf.flags.DEFINE_string("filter_heights", "2,3", "Comma-separated filter heights")
tf.flags.DEFINE_integer("num_filters", 3, "Number of filters per filter size")
tf.flags.DEFINE_string("num_hidden_nodes", "0", "Number of hidden layer's nodes")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Step size of optimizer")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("num_fold", 1, "N fold cross validation")
tf.flags.DEFINE_integer("batch_size", 4, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")

# Tensorflow H/W option
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# print flags
FLAGS = tf.flags.FLAGS          # Make Tensorflow Flag object
FLAGS._parse_flags()            # add flags into "__flags"
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
  print("{}={}".format(attr.upper(), value))
print("")



# Data formatting(loading, change word to index, make N fold)
# ====================================================================

# Load data(reading, parsing, labeling data)
print("Loading data...")
x_sentences, y_type = formatting_data.load_data_and_labels(FLAGS.drive_log_file, FLAGS.net_log_file)
# x_sentences = ["drive doesn't working", "network doens't working", ...]
# y_type      = [1, 0],                   [0, 1]                   , ...]
len_y = len(y_type)

# Build vocabulary(change work to index by dictionary)
max_document_length = max([len(sentence.split(" ")) for sentence in x_sentences])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
sentences_indexs = np.array(list(vocab_processor.fit_transform(x_sentences)))
# 'learn.preprocessing.VocablularyProcessor.fit_transform' change word to index
# e.g. When max_document_length == 4
# ["I like pizza", "i don't like pasta", ..] => [[1, 2, 3, 0] [1, 4, 2, 5], ..]

# make N fold(shuffle, split N fold)
if FLAGS.num_fold <= 0:
  print("Number of fold should be larger than 0")
  sys.exit()
for N in range(FLAGS.num_fold):
  print("\n========================================")
  print("Doing {} fold".format(N+1))
  x_train, x_val, y_train, y_val = formatting_data.make_N_fold(sentences_indexs, y_type, FLAGS.num_fold)
# x_train, x_val, y_train, y_val == numpy.array

# Generate batches(training set)
  batches = formatting_data.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
# make generater : split data by batch_size, do as much as num_epochs on same data


# Training
# ====================================================================
  with tf.Graph().as_default():
    # set session conf
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      # Make CNN model
      cnn = TextCNN(
        max_sentence_length=x_train.shape[1],
        num_hidden_nodes=list(map(int, FLAGS.num_hidden_nodes.split(","))),
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        word_vector_size=FLAGS.word_vector_size,
        filter_heights=list(map(int, FLAGS.filter_heights.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

      # Define Traning procedure
      """
      set optimizer, calculate gradient
      """
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(cnn.loss)
      # loss : RMSE value + L2 regularization
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

      for batch in batches:
        x_batch, y_batch = zip(*batch)

        sess.run(tf.initialize_all_variables())
        
        
        pool_drop, scores, predictions, a, accuracy  = sess.run([cnn.pool_drop, cnn.scores, cnn.predictions, cnn.a, cnn.accuracy], feed_dict = {cnn.input_x : x_batch, cnn.input_y : y_batch, cnn.dropout_keep_prob : FLAGS.dropout_keep_prob})
        print("pool_drop = {}\n\n scores = {}\n\n predictions = {} \n\n a = {}\n\n accuracy = {}".format(pool_drop, scores, predictions, a, accuracy))
        print(cnn.height)
