#! /usr/bin/python3

import re
import os
import glob
import numpy as np
import sys
from refine import *


# flag
in_path = "/root/SMART/ST4000DM000"
#in_path = "/root/SMART/test"
out_directory = "/root/SMART/out_test"
early_raw_filename = os.path.join(out_directory, "early_raw.csv")
middle_raw_filename = os.path.join(out_directory, "middle_raw.csv")
last_raw_filename = os.path.join(out_directory, "last_raw.csv")
failure_raw_filename = os.path.join(out_directory, "failure_raw.csv")

early_nor_filename = os.path.join(out_directory, "early_nor.csv")
middle_nor_filename = os.path.join(out_directory, "middle_nor.csv")
last_nor_filename = os.path.join(out_directory, "last_nor.csv")
failure_nor_filename = os.path.join(out_directory, "failure_nor.csv")
nor_filename = os.path.join(out_directory, "nor.csv")
f_nor_early = open(early_nor_filename, 'w')
f_nor_middle = open(middle_nor_filename, 'w')
f_nor_last = open(last_nor_filename, 'w')
f_nor_failure = open(failure_nor_filename, 'w')
f_nor = open(nor_filename, 'w')

if len(sys.argv) > 1:
  window_size = (int(sys.argv[1]))
else:
  window_size = 10
print("working...")
i = 0
# open all file
for in_filename in glob.glob(os.path.abspath(os.path.join(in_path, '*.csv'))):

# 1. find serial number of product which have the failure
  cur_window_size = window_size
  # check last line if there are failure.
  with open(in_filename, 'r') as in_f:
    lines = in_f.readlines()
    max_window_size = len(lines) - 1
    
    if  cur_window_size > max_window_size:
      cur_window_size = max_window_size
    last_line = lines[-1]
    str_fail_flag = last_line.split(',')[4]
  # if not, close and check next file.
    if str_fail_flag == '0':
      continue

# 2. extract previous 'window_size' lines before failure, and classify(raw, normalized)
    _, lines_filted_attr = select_attribute([3, 5, 7, 9, 187, 188, 197, 198], lines)
    raw_lines, nor_lines = classify_attribute2(lines_filted_attr)

# 3. save as cluster
    num_line = len(nor_lines)
    num_len = 2
    early_end = num_len + 1
    middle_start = int(num_line / 2)
    middle_end = middle_start + num_len
    last_start = -(num_len + 1)
    if i == 0:
      f_nor_early.writelines(nor_lines[:early_end])
      f_nor_middle.writelines(nor_lines[0])
      f_nor_middle.writelines(nor_lines[middle_start:middle_end])
      f_nor_last.writelines(nor_lines[0])
      f_nor_last.writelines(nor_lines[last_start:-1])
      f_nor_failure.writelines(nor_lines[0])
      f_nor_failure.writelines(nor_lines[-1])
      # nor.write
      f_nor.writelines(nor_lines[:early_end])
      f_nor.writelines(nor_lines[middle_start:middle_end])
      f_nor.writelines(nor_lines[last_start:])
      i = i + 1
    else:
      f_nor_early.writelines(nor_lines[1:early_end])
      f_nor_middle.writelines(nor_lines[middle_start:middle_end])
      f_nor_last.writelines(nor_lines[last_start:-1])
      f_nor_failure.writelines(nor_lines[-1])
      # nor.write
      f_nor.writelines(nor_lines[1:early_end])
      f_nor.writelines(nor_lines[middle_start:middle_end])
      f_nor.writelines(nor_lines[last_start:])

print("complete!")
