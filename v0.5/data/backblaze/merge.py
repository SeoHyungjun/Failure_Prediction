import csv
import os
from random import *

path_dir = '/Failure_Prediction/v0.4/data/backblaze/merge'
file_list = os.listdir(path_dir)
file_list.sort()

print(file_list)
