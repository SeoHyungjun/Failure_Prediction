#! /usr/bin/python3

import glob
import numpy as np
import os
import sys

from refine import *

"""
split fail, non_fail serial
"""
# FLAG
IN_PATH = "/root/SMART/ST4000DM000"
#IN_PATH = "/root/SMART/test"
OUT_FAIL_DIR = "/root/SMART/in_ann/fail_serial"
OUT_NON_FAIL_DIR = "/root/SMART/in_ann/non_fail_serial"
ATTRIBUTES = [3,5,7,9,187, 188, 197, 198]
if len(sys.argv) > 1:
    WINDOW_SIZE = (int(sys.argv[1]))
else:
    WINDOW_SIZE = 10
print("working...")


num_failure_serial = 0
num_non_failure_serial = 0
# open all file
for in_filepath in glob.glob(os.path.abspath(os.path.join(IN_PATH, '*.csv'))):
    with open(in_filepath, "r") as in_f:
        lines = in_f.readlines()
        max_window_size = len(lines) - 1
        if max_window_size < WINDOW_SIZE:
            cur_window_size = max_window_size
        else:
            cur_window_size = WINDOW_SIZE        
        # 1. check whether there are failure. keep balance between fail and non-fail file number.
        if len(lines) < 10:
            os.remove(in_filepath)
            continue
        last_line = lines[-1]
        str_fail_flag = last_line.split(',')[4]
        if str_fail_flag == '0':   # non-failure
            if num_failure_serial < num_non_failure_serial:
                continue
            num_non_failure_serial = num_non_failure_serial + 1
        elif str_fail_flag == '1': # failure
            num_failure_serial = num_failure_serial + 1

        # 2. get specific normarlized attriube 
        _, lines_filted_attr = select_attribute(ATTRIBUTES, lines)
        _, nor_lines = classify_attribute2(lines_filted_attr)

        # 3. save
        filename = in_filepath.split('/')[-1]
        if str_fail_flag == '0':
            f_non_fail = open(os.path.join(OUT_NON_FAIL_DIR, filename), "w")
            f_non_fail.writelines(nor_lines)
        elif str_fail_flag == '1':
            f_fail = open(os.path.join(OUT_FAIL_DIR, filename), "w")
            f_fail.writelines(nor_lines)

