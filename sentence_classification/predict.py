#! /usr/bin/python3

import numpy as np
import os
import datetime
import csv

import tensorflow as tf
from tensorflow.contrib import learn

import formatting_data 
from text_cnn import TextCNN


# Flags
# ==============================================================

# Data loading
tf.flags.DEFINE_string("drive_log_file", "./data/dict/disk", "Data source for the drive log file")
tf.flags.DEFINE_string("net_log_file", "./data/dict/net", "Data source for the net log file")

# Predict parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Model checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Tensorflow H/W option
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
  x_raw, y_test = formatting_data.load_data_and_labels(FLAGS.drive_log_file , FLAGS.net_log_file)
  y_test = np.argmax(y_test, axis=1)
else:
  x_raw = []
  print("Enter the sentence")
  x_raw.append(input())
  y_test = None
#  x_raw = ["scsi error", "ip doesn't match", "error"]
#  y_test = [0, 1, 1]

# Load vocabularay index, and initiate vocabulary processor object
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
# x_test consist of word index

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
  session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
  sess = tf.Session(config=session_conf)
  with sess.as_default():
  ### 1. Load the saved 'meta graph' and restore 'variables'
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

  ### 2. get the operation
    # Get the placeholders from the graph by name
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    # input_y = graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    # Tensors we want to evaluate
    predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    softmax_scores = graph.get_operation_by_name("output/softmax_scores").outputs[0]
    # Generate batches for one epoch
    batches = formatting_data.batch_iter(list(x_test), FLAGS.batch_size, 1)
    # Collect the predictions here
    all_predictions = []

    for x_test_batch in batches:
      batch_predictions, batch_softmax_scores = sess.run([predictions, softmax_scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
      all_predictions = np.concatenate([all_predictions, batch_predictions])
#      print("all_predictions ={}".format(all_predictions))
#      print("all softmax ={}".format(all_softmax_scores))
      if 'all_softmax_scores' in locals():
        all_softmax_scores = np.concatenate([all_softmax_scores, batch_softmax_scores])
      else :
        all_softmax_scores = batch_softmax_scores

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions, all_softmax_scores))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
  csv.writer(f).writerows(predictions_human_readable)
