#! /usr/bin/python3

import os
import glob
import numpy as np

import pandas as pd
import tensorflow as tf

from k_means import *

"""
per serial, make two ann input(in first and last line).
"""


def get_first(data, window_size, prediction_interval):
    first_line = data[0:window_size]
    return first_line

def get_last(data, window_size, prediction_interval):
    last_line = data[-(window_size+prediction_interval):-prediction_interval]
    return last_line


# Flag
IN_FAILURE_PATH = "/root/SMART/in_ann/fail_serial"
IN_NON_FAILURE_PATH = "/root/SMART/in_ann/non_fail_serial"
OUT_FILE_PATH = "/root/SMART/in_ann/in_ann.csv"
WINDOW_SIZE = 6
PREDICTION_INTERVAL = 3
INT_FAILURE = 0
INT_NON_FAILURE = 1
index = 0

os.remove(OUT_FILE_PATH)
f_out = open(OUT_FILE_PATH, "a")

# write data that have failure. two data added(initial one, last one-failure)
for in_filepath in glob.glob(os.path.abspath(os.path.join(IN_FAILURE_PATH, '*.csv'))):
    df_csv = pd.read_csv(in_filepath)
    col_failure = df_csv['failure']
    df_csv = df_csv.drop('failure', 1)
    with tf.Session() as sess:
        k_means = K_Means(session=sess)
        k_means.restore_all()
        result = k_means.run(df_csv)

    first_line = get_first(result, WINDOW_SIZE, PREDICTION_INTERVAL)
    first_line = np.append(first_line, INT_NON_FAILURE)
    str_first_line = ""
    for i, x in enumerate(first_line):
        if i != len(first_line) - 1:
            str_first_line = str_first_line + str(x) + ','
        else:
            str_first_line = str_first_line + str(x) + '\n'

    last_line = get_last(result, WINDOW_SIZE, PREDICTION_INTERVAL)
    last_line = np.append(last_line, INT_FAILURE)
    str_last_line = ""
    for i, x in enumerate(last_line):
        if i != len(last_line) - 1:
            str_last_line = str_last_line + str(x) + ','
        else:
            str_last_line = str_last_line + str(x) + '\n'

    lines = [str_first_line, str_last_line]
    f_out.writelines(lines)

# write data that don't have failure. two data added(first one, last one)
for in_filepath in glob.glob(os.path.abspath(os.path.join(IN_NON_FAILURE_PATH, '*.csv'))):
    df_csv = pd.read_csv(in_filepath)
    col_failure = df_csv['failure']
    df_csv = df_csv.drop('failure', 1)
    with tf.Session() as sess:
        k_means = K_Means(session=sess)
        k_means.restore_all()
        result = k_means.run(df_csv)

    first_line = get_first(result, WINDOW_SIZE, PREDICTION_INTERVAL)
    first_line = np.append(first_line, INT_NON_FAILURE)
    str_first_line = ""
    for i, x in enumerate(first_line):
        if i != len(first_line) - 1:
            str_first_line = str_first_line + str(x) + ','
        else:
            str_first_line = str_first_line + str(x) + '\n'

    last_line = get_last(result, WINDOW_SIZE, PREDICTION_INTERVAL)
    last_line = np.append(last_line, INT_NON_FAILURE)
    str_last_line = ""
    for i, x in enumerate(last_line):
        if i != len(last_line) - 1:
            str_last_line = str_last_line + str(x) + ','
        else:
            str_last_line = str_last_line + str(x) + '\n'

    lines = [str_first_line, str_last_line]
    f_out.writelines(lines)
