import numpy as np
import os
import re


def select_attribute(index_attributes, lines):
    pass


def exclude_empty(lines):
    """
    exclude empty attribute by inspecting last line.
    input : string lines seperated by comma.
    """
    filtered_index = list()
    filtered_lines = list()
    last_line = lines[-1].split(',')
    for i, content in enumerate(last_line):
        if content != '':
            filtered_index.append(i)
    for line in lines:
        l = np.array(line.split(','))
        l = l[filtered_index]
        filtered_line.append(i)
    return filtered_index, filtred_line


def filter_attribute(line_attributes, lines):
  """
  step 1. filter empty attribute
  step 2. filter unchanged attribute
  input : csv format lines -> output : filtered csv format lines
  """
  filtered_index1 = list()
  filtered_lines1 = list()
  # step 1. filter empty attr index and make list
  filtered_index1, filtered_lines1 = exclude_empty(lines)

  # step 2. filter no-changed attr index
  filtered_index2 = list()
  filtered_lines2 = list()
  num_lines = len(lines)
  num_attributes = len(filtered_index1)
  pre_value = ''
  flag_changed = False
  for index_attr in range(num_attributes):
    pre_value = filtered_lines1[0][index_attr]
    if index_attr == 0:  # exclude date attribute
      flag_changed = False
    else:
      for index_line in range(num_lines-1):
        if pre_value != filtered_lines1[index_line+1][index_attr]:
          flag_changed = True
        pre_value = filtered_lines1[index_line+1][index_attr]

    if flag_changed == True:
      filtered_index2.append(index_attr)
      flag_changed = False

  # fix attribute name
  line_attributes = re.sub("smart_", "", line_attributes)
  line_attributes = re.sub(r"\n", "", line_attributes)
  attributes = np.array(line_attributes.split(','))
  attributes = attributes[filtered_index1]
  attributes = attributes[filtered_index2]
  attributes = ",".join(attributes) + '\n'
  filtered_lines2.append(attributes)

  # extract smart value and add to output
  for line in filtered_lines1:
    l = np.array(line)
    l = l[filtered_index2]
    l = ",".join(l)
    l = re.sub(r"\n", "", l) + '\n'
    filtered_lines2.append(l)

  return filtered_lines2


def classify_attribute(lines):
  """
  classify raw and normalized data
  input : csv format lines -> output : filtered csv format lines
  """

  raw_index = list([])
  nor_index = list([])

  raw_lines = list()
  nor_lines = list()

  attributes = lines[0].split(',')
  for i, attribute in enumerate(attributes):
    if "failure" in attribute:
      raw_index.append(i)
      nor_index.append(i)
    if i > 5:
      break

  print(raw_index)
  for i, attribute in enumerate(attributes):
    if "raw" in attribute:
      raw_index.append(i)
    elif "normalized" in attribute:
      nor_index.append(i)

  for i, line in enumerate(lines):
    raw_line = np.array(line.split(','))
    raw_line = raw_line[raw_index]
    raw_line = ",".join(raw_line)
    raw_line = re.sub(r"\n", "", raw_line) + '\n'
    raw_lines.append(raw_line)

    nor_line = np.array(line.split(','))
    nor_line = nor_line[nor_index]
    nor_line = ",".join(nor_line)
    nor_line = re.sub(r"\n", "", nor_line) + '\n'
    nor_lines.append(nor_line)

  return raw_lines, nor_lines
