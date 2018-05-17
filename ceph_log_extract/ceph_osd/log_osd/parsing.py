#! /usr/bin/python3

import re

filepath = "./"
filenames = ["ldout", "dout", "lderr"]
outfile_tag = "_parsed.csv"
key = ['filename', 'func', 'context', 'line', 'message']

for filename in filenames:
    full_filepath = filepath+filename
    full_outfilepath = full_filepath+outfile_tag
    with open(full_filepath, 'r') as infile, open(full_outfilepath, 'w') as outfile:
        for index, line in enumerate(infile):
            if index == 0 or index == 1:
                continue
            print(index, line)
            words = re.split('\s', line)
            value = list()
            for word in words:
                if word != '':
                    key.append(word)

            print(key)
        exit(0)


