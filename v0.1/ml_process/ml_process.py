#!/bin/python3

# import sys
# sys.path.insert(0, '../library/data_transform')
# sys.path.insert(0, '../library/model')
# import eval_info
# import cnn
# import library
from abc import ABCMeta
from abc import abstractmethod

class ml_process_main :
    def __init__(self):
        self.algo_num = 0
        self.model_list = []
        self.transform_list = []

    # Configuration can be operated by "UI interaction" or "CLI interaction"
    # Configuration information can be stored in "DB" or "just config file"
    # But config file should be existed. you can configure in the config filel
    # where you read configuration information (DB or file).
    @abstractmethod
    def config(self, config_fname):
        pass

    # config_parse is called by config.
    # Configuration variables are set in this function.
    @abstractmethod
    def config_parse(self, config_fname):
        pass

    # read_where can be 'db' or 'pipe' or 'queue' ...? 
    @abstractmethod
    def read_data(self, data_read_where):

    @abstractmethod
    def main(self):
        pass


