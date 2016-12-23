import numpy as np
import re
import sys

def clean_str(string):
  """
  filtering, substitute specific character from string.
  """
  # strip() is filtering white space character(e.g. '\n')
  # re.sub() substitude specific character
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) #replace special chars
  string = re.sub(r"\'s", " \'s", string)                # It's    => It 's
  string = re.sub(r"\'ve", " \'ve", string)              # They've => They 've
  string = re.sub(r"n\'t", " n\'t", string)              # don't   => don 't
  string = re.sub(r"\'re", " \'re", string)              # you're  => you 're
  string = re.sub(r"\'d", " \'d", string)                # she'd   => she 'd
  string = re.sub(r"\'ll", " \'ll", string)              # he'll   => he 'll
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()
  # "\s{n, m}" mean n~m white space.
  # "\s{2,} means more than 2 white space.


def load_data_and_labels(drive_log_file, net_log_file):
  """
  Load log file, split sentences into [ [S1, S2, ..], [[L1], [L2], ..] ]
  as following
  [ ['drive doesn't working', 'network doens't working', ...]  [[1, 0], [0, 1],,,] ]
  """

  # Load data from files
  drive_sentences = open(drive_log_file, "r").readlines()
  net_sentences = open(net_log_file, "r").readlines()

  # Split by words and parsing
  x_sentences = drive_sentences + net_sentences
  x_sentences = [clean_str(sentence) for sentence in x_sentences]

  # Generate labels
  drive_labels = [[1, 0] for _ in drive_sentences]
  net_labels =  [[0, 1] for _ in net_sentences]
  y_type = np.concatenate([drive_labels, net_labels], 0)

  return [x_sentences, y_type]


def make_N_fold(sentences_indexs, y_type, num_fold):
  """
  Make N fold for 'N fold cross validation'
  """
  len_y = len(y_type)
  if len_y < 2: 
    print("Too small data set. Need more data... exit.")
    sys.exit()

  # Randomly shuffle data
  np.random.seed()
  shuffle_indices = np.random.permutation(np.arange(len_y))
  # shuffle_indices = [3, 2, 8, 10, 38, 1, ...]
  x_shuffled = sentences_indexs[shuffle_indices]
  y_shuffled = y_type[shuffle_indices]

  # Split train/test set
  if num_fold <= 0:
    print("A")
    print("Number of fold should be larger than 0")
    sys.exit()
  elif num_fold == 1:
    val_sample_percentage = 0
    # 1 fold mean 'all data set used as training set'
  else:
    val_sample_percentage = float(1) / float(num_fold)

  val_sample_index = -1 * int(val_sample_percentage * float(len_y))
  if val_sample_index == 0: 
    val_sample_index = -1
    # even though data set or val_sample_percentage is small,
    # val_data is more than one
  x_train, x_val = x_shuffled[:val_sample_index], x_shuffled[val_sample_index:]
  y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]
  return x_train, x_val, y_train, y_val


def batch_iter(data, batch_size, num_epochs):
  """
  slice data set by batch_size and repeat as num_epochs
  for learning as much times as num_epochs with same data, by batch_size
  """
  data_size = len(data)
  if data_size % batch_size == 0:
    num_batches_per_epoch = int(data_size/batch_size)
  else:
    num_batches_per_epoch = int(data_size/batch_size) + 1
  for epoch in range(num_epochs):
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min(start_index + batch_size, data_size)
      yield data[start_index:end_index]
  # A[n:m] => not contain A[m]
  # even though start_index is out of list's range. if end_index <= len(list), 'out of range' error doesn't occur. data[start_index:end_index]
    print("End \'{}\' epoch\n".format(epoch + 1))
