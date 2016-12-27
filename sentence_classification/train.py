#! /usr/bin/env python3 

import sys
import os
import datetime

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import formatting_data
from text_cnn import TextCNN


# Parameters(set flag. print flag)
# ====================================================================

# Data loading
tf.flags.DEFINE_string("drive_log_file", "./data/dict/disk", "Data source for the drive log file")
tf.flags.DEFINE_string("net_log_file", "./data/dict/net", "Data source for the net log file")

# Model Hyperparameters
tf.flags.DEFINE_integer("word_vector_length", 128, "lenth of each word vector")
tf.flags.DEFINE_string("filter_heights", "3,4,5", "Comma-separated filter heights")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size")
tf.flags.DEFINE_string("num_hidden_nodes", "0", "Number of hidden layer's nodes")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Step size of optimizer")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("num_fold", 10, "N fold cross validation")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs")
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


# Configure Training
# ====================================================================
  with tf.Graph().as_default():
    # set session conf
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
    ### 1. Make CNN model
      cnn = TextCNN(
        max_sentence_length=x_train.shape[1],
        num_hidden_nodes=list(map(int, FLAGS.num_hidden_nodes.split(","))),
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        word_vector_length=FLAGS.word_vector_length,
        filter_heights=list(map(int, FLAGS.filter_heights.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

    ### 2. Define Traning procedure
      """
      set optimizer, calculate gradient
      """
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(cnn.loss)
      # grads_and_vars have gradient when each value is changed.
      # value mean (trainable == true)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


   ### 3. summary
      # (Optional) ~
      grad_summaries = []
      for g, v in grads_and_vars:
        if g is not None:
          grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
          sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
          grad_summaries.append(grad_hist_summary)
          grad_summaries.append(sparsity_summary)
      grad_summaries_merged = tf.merge_summary(grad_summaries)
      # ~ (Optional)
      loss_summary = tf.scalar_summary("loss", cnn.loss)
      accuracy_summary = tf.scalar_summary("accuracy", cnn.accuracy)
      # Train, Dev
      train_summary_op = tf.merge_summary([loss_summary, accuracy_summary, grad_summaries_merged])
      dev_summary_op = tf.merge_summary([loss_summary, accuracy_summary])


   ### 4. set out directory
      timestamp = datetime.datetime.now().strftime("%m'%d(%H:%M)")
      out_dir = os.path.join(os.path.expanduser("~"), "Desktop/textCNN_result", timestamp)
      print("Writing to {}\n".format(out_dir))
      train_summary_dir = os.path.join(out_dir, "summaries", "train")
      dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
      flag_conf = os.path.join(out_dir, "flag.conf")
      accuracy_result = os.path.join(out_dir, "accuracy_result")
      # checkpoint(where model is saved)
      checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
      checkpoint_prefix = os.path.join(checkpoint_dir, "model")
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

   ### 5. saver(operation) and writer. writer is class
      train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
      dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
      saver = tf.train.Saver(tf.all_variables())

   ### 6. save vocabulary index dictionary, save flags
      vocab_processor.save(os.path.join(out_dir,"vocab"))
      with open(flag_conf, "w") as fd:
        for attr, value in sorted(FLAGS.__flags.items()):
          str_flag = "{} = {}".format(attr.upper(), value)
          fd.write(str_flag + "\n")

# Training
# ====================================================================

    ### 1. defining one step
      def train_step(x_batch, y_batch, writer=None):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x : x_batch,
          cnn.input_y : y_batch,
          cnn.dropout_keep_prob : FLAGS.dropout_keep_prob
        }
        # should contain 'train_op' in sess.run() agrument => increase global step
        _, step, summaries, loss, accuracy = sess.run(
          [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
          feed_dict)
        time_str = datetime.datetime.now().strftime("%m'%d(%H:%M:%S)")
        #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
          writer.add_summary(summaries, step)

      def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x : x_batch,
          cnn.input_y : y_batch,
          cnn.dropout_keep_prob : 1.0
        }
        step, summaries, loss, accuracy = sess.run(
          [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
          feed_dict)
        time_str = datetime.datetime.now().strftime("%m'%d(%H:%M:%S)")
        str_eval = "{}: step {}, loss{:g}, acc {:g}".format(time_str, step, loss, accuracy)
        print(str_eval)
        if writer:
          writer.add_summary(summaries, step)
        return str_eval

    ### 2. training batch by batch. as much as epochs with same data
      sess.run(tf.initialize_all_variables())
      for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch, writer=train_summary_writer)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
          print("Evaluation:")
          eval_result = dev_step(x_val, y_val, writer=dev_summary_writer)
          fd_accuracy = open(accuracy_result, "a")
          fd_accuracy.write(eval_result + "\n")
          fd_accuracy.close()
        if current_step % FLAGS.checkpoint_every == 0:
          path = saver.save(sess, checkpoint_prefix, global_step=current_step)
          print("Saved model to {}\n".format(path))

        
        """
        pool_drop, scores, predictions, accuracy  = sess.run([cnn.pool_drop, cnn.scores, cnn.predictions, cnn.accuracy], feed_dict = {cnn.input_x : x_batch, cnn.input_y : y_batch, cnn.dropout_keep_prob : FLAGS.dropout_keep_prob})
        print("pool_drop = {}\n\n scores = {}\n\n predictions = {} \n\n accuracy = {}".format(pool_drop, scores, predictions, accuracy))
        """
