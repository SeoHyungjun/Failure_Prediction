import tensorflow as tf
import numpy as np


class TextCNN(object):
  """ 
  A CNN for text classification.
  Uses an embedding layer, followed by a convolutional, max-pooling, hidden layer and softmax layer.
  """
  def __init__(
    self, max_sentence_length, num_classes, vocab_size,
    word_vector_length, filter_heights, num_filters, l2_reg_lambda=0.0):

    # Placeholders for input, output and dropout(input:sentences, output:classes)
    self.input_x = tf.placeholder(tf.int32, [None, max_sentence_length], name="input_x")
    # e.g. input_x == [ [1,2,3,0], [2,3,1,6], ... ], consist of word index
    self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)


## ============================================================= ##

    # 1. Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      W = tf.Variable(
        tf.random_uniform([vocab_size, word_vector_length], -1.0, 1.0), name="W")
      # 'W' consist of word vector which is matched by word index
      self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
      self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) 
      # e.g. embedded_chars == [ [[0.1 0.2 0.4] [0.5 0.6 0.1] ...] [[..].. ]..]
      # [[0.1 0.2 0.4] [0.5 0.6 0.1] ...] is one sentence


## ============================================================= ##

    # 2. Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_height in enumerate(filter_heights):
      with tf.name_scope("conv-maxpool-%s" % filter_height):
        # Convolution Layer
        filter_shape = [filter_height, word_vector_length, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
          self.embedded_chars_expanded,
          W,
          strides=[1,1,1,1],
          padding="VALID",
          name="conv")
        # Apply nonlinearity
        conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
          conv_relu,
          ksize=[1, max_sentence_length - filter_height + 1, 1, 1],
          strides=[1,1,1,1],  # ksize : maxpool window
          padding="VALID",
          name="pool")
        pooled_outputs.append(pooled)


    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    self.pooled_result = tf.concat(3, pooled_outputs)
    self.pooled_result_flat = tf.reshape(self.pooled_result, [-1, num_filters_total])
    # pooled_result == [[[ [sentence 1 features] [sentence 2 features] .. ]]]
    # e.g. pooled_result == [[[ [0.3 1.4 7.8 ...] [9.5 8.3 ...] .... ]]]
    # features collected by all filters of variety sizes

    # Add dropout
    with tf.name_scope("pool_dropout"):
      self.pool_drop = tf.nn.dropout(self.pooled_result_flat, self.dropout_keep_prob)


## ============================================================= ##

    # 3. Hidden_NN layer
    pre_num_node = num_filters_total
    self.NN_result = [None] * (len(num_nodes) + 1)
    self.NN_result[0] = self.pool_drop
    for index, num_node in enumerate(num_nodes):
      if num_node == 0:
        index= -1
        break
      with tf.name_scope("Hidden_NN{}".format(index+1)):
        num_nodes = num_node
        W = tf.get_variable(
          "W",
          shape = [pre_num_node, num_node],
          initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_node]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.NN_result[index+1] = tf.sigmoid(tf.nn.xw_plus_b(self.NN_result[index], W, b))
        # Add dropout
        with tf.name_scope("NN_dropout"):
          self.NN_result[index+1] = tf.nn.dropout(self.NN_result[index+1], self.dropout_keep_prob)
        pre_num_node = num_node


## ============================================================= ##

    # 4. output node(scores, predictions)
    with tf.name_scope("output"):
      W = tf.get_variable(
        "W",
        shape = [pre_num_node, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)
      # tf.nn.l2_loss(a) = sum(a^2)/2, element-wise
      self.scores = tf.nn.xw_plus_b(self.NN_result[index+1], W, b, name="scores")
      self.predictions = tf.argmax(self.scores, 1, name="predictions")


## ============================================================= ##

    #  5. loss(entropy) and Accuracy
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
      self.loss = tf.reduce_mean(losses) + (l2_reg_lambda * l2_loss)

    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
      # e.g. correct_predictions = [ True, Ture, False, ....]
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
      
## ============================================================= ##
