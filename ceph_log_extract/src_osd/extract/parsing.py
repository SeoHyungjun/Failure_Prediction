#! /usr/bin/python3

import re
import csv

filepath = "./"
filenames = ["dout", "ldout", "lderr"]
outfile_tag = "_parsed.csv"
keys_str = ['tag','filename','func','line','message']

log_dict = dict()

for filename in filenames:
    full_filepath = filepath+filename
    tag = filename
    full_outfilepath = full_filepath+outfile_tag
    # open file each log-relevant function
    with open(full_filepath, 'r') as infile, open(full_outfilepath, 'w') as outfile:
        csvwriter = csv.writer(outfile, lineterminator='\n')
        csvwriter.writerow(keys_str)
#        outfile.write(keys_str)
        for index, line in enumerate(infile):
            if index == 0 or index == 1:
                continue
            # filename , line, func...
            elif index % 2 == 0:
                words = list()
                words_with_space = re.split('\s', line)
                for word in words_with_space:
                    if word != '':
                        words.append(word)
                values = list()
                values.append(tag)
                values.append(words[2])   # filename containing that func
                values.append(words[3])   # func calling tag
                values.append(words[1])   # line number
            # code
            elif index % 2 == 1:
                msg = ''
                words_with_space = re.split('<<', line)
                for word in words_with_space:
                    word = word.strip()
                    if word != 'dendl;' and tag not in word:
                        msg = msg + ' ' + word
                values.append(msg)
#                values.append('\n')
#                outfile.writelines(values)
                csvwriter.writerow(values)
                print(values)
