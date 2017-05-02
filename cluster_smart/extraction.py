#! /usr/bin/python3

import re
import os
import glob
import numpy as np
import sys
from refine import *


# flag
#in_path = "ST4000DM000"
in_path = "test2"
out_directory = "out_test"
if len(sys.argv) > 1:
  window_size = (int(sys.argv[1]))
else:
  window_size = 10
print("working...")

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
    extracted_lines = filter_attribute(lines[0], lines[-(cur_window_size):])
    raw_lines, nor_lines = classify_attribute(extracted_lines)

# 3. save each file per serial number
    filename = re.sub(".csv", "", in_filename.split('/')[-1])
    out_raw_filename = os.path.join(out_directory, filename + "_raw.csv")
    out_nor_filename = os.path.join(out_directory, filename + "_normalized.csv")
    with open(out_raw_filename, 'w') as out_f:
      out_f.writelines(raw_lines)
    with open(out_nor_filename, 'w') as out_f:
      out_f.writelines(nor_lines)

print("complete! well done")
