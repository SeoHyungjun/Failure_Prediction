import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      
        # Keeping track of l2 regularization loss (optional)
        # TODO : Don't know well
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(	
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
		# code interpret #
		# make random number in format [vocab_size, embedding_size] between -1 and 1
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # code interpret #
            # guess W is "word imbedding dictionary", and learned during training
            # ex) W = [voca1, voca2, ..., vocaN] = [[1, 0, 2, 3], [1, 2, 1, 1] ...]
            # guess input_x is "index" in "word imbedding dictionary"
            # so embedded_chars is result of each word embedded by word index.
            # if one sentence1 composed of "I am a boy" and each index is (3, 100, 1, 4), 
            # ex) embedded_chars = [sentence1, sentence2, ...] = [[voca3, voca100, voca1, voca4], ....]

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []   
	# code interpret #
	# collect all pooled outptut(by filter1, filter2, ..)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)  # the number of all filters
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
	    # code interpret #
            # score = XW + B, maybe the number of filter = 384 
            # X = shape(the number of sentence*384)
            # W = shape(384*2)
            # scores = shape(the number of sentence * 2)
            #        = [[probability value that sentence1 is class1, probability value that sentence1 is class2], [p(S2->1), p(S2->2)], .....], not percent, just value

            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # code interpret #
            # return index of individual row that have highest value
            # ex) predictions = [0, 1, 1, 0, 1, .....]
            self.softmax_scores = tf.nn.softmax(self.scores, -1, name="softmax_scores")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            # losses = [loss of sentence 1(float), loss of sentence 2(float), ....]
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
