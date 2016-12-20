import numpy as np
import re


def clean_str(string):
  """
  filtering, substitute specific character from string.
  """
  # strip() is filtering white space character(e.g. '\n')
  # re.sub() substitude specific character
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) #remove special chars
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
  Load log file, split sentences into [ ['S1', 'S2', ..], [[L1], [L2], ..] ]
  as following
  [ ['drive doesn't working', 'network doens't working', ...]  [[1, 0], [0, 1],,,] ]
  """

  # Load data from files
  drive_sentences = open(drive_log_file, "r").readlines()
  net_sentences = open(net_log_file, "r").readlines()

  drive_sentences = [clean_str(s) for s in drive_sentences]

  print(drive_sentences)

  print(re.sub(r"\w", "\"", "aasdfasdfjasdl'"))

  return 1, 2

