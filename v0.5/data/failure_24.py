import csv
import os
from random import *

path_dir = '/Failure_Prediction/v0.4/data/failure'
file_list = os.listdir(path_dir)
file_list.sort()

for file_name in file_list:
    count = 0
    with open('failure/' + file_name, 'r') as f:
        lines = list(csv.reader(f))

    with open('h_failure/failure_24.csv', 'a') as f:
        for line in lines:
            if count >= len(lines) - 24:
                for i in range(0,14):
                    if i != 13:
                        f.write(str(line[i]) + ',')
                    else:
                        f.write(str(line[i]) + '\n')
            count = count + 1
