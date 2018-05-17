#! /usr/bin/python3

import re
import csv

FILEPATH = "./"
FILENAMES = ["ldout", "dout", "lderr"]
OUTFILE_TAG = "_parsed.csv"
KEYS_STR = ['tag','filename','func','line','message']
FLAG_REPLACE_FUNC_IN_MSG = True

log_dict = dict()

for filename in FILENAMES:
    full_filepath = FILEPATH+filename
    tag = filename
    full_outfilepath = full_filepath+OUTFILE_TAG
    # open file each log-relevant function
    with open(full_filepath, 'r') as infile, open(full_outfilepath, 'w') as outfile:
        csvwriter = csv.writer(outfile, lineterminator='\n')
        csvwriter.writerow(KEYS_STR)
#        outfile.write(KEYS_STR)
        for index, line in enumerate(infile):
            # tag information
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
                values.append(',')
                values.append(words[2])   # filename containing that func
                values.append(',')
                values.append(words[3])   # func calling tag
                values.append(',')
                values.append(words[1])   # line number
                values.append(',')
            # code
            elif index % 2 == 1:
                msg = ''
                words_with_space = re.split('<<', line)
                for word in words_with_space:
                    word = word.strip()
                    tag_func_str = tag + '('
                    if word != 'dendl;' and tag_func_str not in word:
                        if FLAG_REPLACE_FUNC_IN_MSG == True and word == '__func__':
                            word = values[4]
                        msg = msg + ' ' + word

                values.append(msg)
                values.append('\n')
                outfile.writelines(values)
                if words_with_space[-1].strip() != 'dendl;': 
                    print(values)
                    with open("deficient_log", 'a') as f:
                        f.writelines(values)
#                    print(repr(words_with_space[-1].strip()))
