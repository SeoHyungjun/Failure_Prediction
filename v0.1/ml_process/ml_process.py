#!/bin/python3

# import sys
# sys.path.insert(0, '../library/data_transform')
# sys.path.insert(0, '../library/model')
# import eval_info
# import cnn
# import library
import configparser as cp
import library

from abc import ABCMeta
from abc import abstractmethod
import model

class ML_process_class :
    def __init__(self, config_fname='config'):
        self.model_num = 0
        self.model_list = []
        self.model_name_list = []
        self.cfg_name = config_fname
        self.lib = library()

    # config reads the config file and config
    # Configuration item will be what you use algorithm and data_transform...
    def config(self, config_fname='config'):
        config = cp.ConfigParser()
        config.read(config_fname)
        self.model_num = int(config['ML_Process']['model_num'])
        self.model_name_list = config['ML_Process']['model_names'] \
                                .replace(' ','').split(',')

        for model in self.model_name_list :
            # algoparam_dict has each algorithm's parameters
            # including train_data_source, train_result_target, predict_data_source

            # for lib_model in self.lib.models :
            #     if lib_model.name == model :

            if model == 'model1' :
                self.lib.model1.param1 = config['model1']['param1']
                self.lib.model1.param2 = config['model1']['param2']
                self.lib.model1.param3 = config['model1']['param3']
                self.lib.model1.train_operations = config['model1']['train_operations']
            elif model == 'model2' :
                self.lib.model2.arg1 = config['model2']['arg1']
                self.lib.model2.arg2 = config['model2']['arg2']
                self.lib.model2.arg3 = config['model2']['arg3']
                self.lib.model2.train_operations = config['model2']['train_operations']


    def print_config(self) :
        print("Configuration information")
        print("{0:20s} : ".format("Config file path") + "%s" % self.cfg_name)
        print("{0:20s} : ".format("Number of Algorithm") + "%d" % self.model_num)
        print("{0:20s} : ".format("Algorithms") + ' '.join(self.model_name_list))


    # read_training_data or predict_data
    # read_where can be 'db' or 'pipe' or 'queue' ...? 
    @abstractmethod
    def read_data(self, data_read_where):
        pass

    @abstractmethod
    def main(self):
        pass


