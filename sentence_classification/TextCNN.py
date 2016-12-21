import tensorflow as tf
import numpy as np


class TextCNN(object):
  """
  CNN model for text classification.
  """
  
  def __init__(
    self, max_sentence_length, num_classes, vocab_size,
    embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
    
    # vocab_size : the number of word in dictionary
    # meaning of embedding : make 'a' word into a vector
    # embedding_size : length of word vector
