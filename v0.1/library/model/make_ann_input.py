#! /usr/bin/python3

import os
import glob

import pandas as pd
import tensorflow as tf

from k_means import *

"""
per serial, make two ann input(in first and last line).
"""


# Flag
IN_FAILURE_PATH = "/root/SMART/in_ann/fail_serial"
IN_NON_FAILURE_PATH = "/root/SMART/in_ann/non_fail_serial"

flag = 0
for in_filepath in glob.glob(os.path.abspath(os.path.join(IN_NON_FAILURE_PATH, '*.csv'))):
    df_csv = pd.read_csv(in_filepath)
    col_failure = df_csv['failure']
    df_csv = df_csv.drop('failure', 1)

    with tf.Session() as sess:
        k_means = K_Means(session=sess)
        k_means.restore_all()
        result = k_means.run(df_csv)

    flag = flag + 1
    if flag == 6:
        break

    # 1. remove fail attribute. cluster run.

    # 2. attach fail attribute.
