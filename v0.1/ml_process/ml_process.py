#!/bin/python3
#-*- coding:utf-8 -*-

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
        self.lib = library.library()

    # config reads the config file and config
    # Configuration item will be what you use algorithm and data_transform...
    def config(self, config_fname='config'):
        config = cp.ConfigParser()
        config.read(config_fname)
        self.model_num = int(config['ML_Process']['model_num'])
        self.model_name_list = config['ML_Process']['model_names'] \
                                .replace(' ','').split(',')

        for model_name in self.model_name_list :
            # algoparam_dict has each algorithm's parameters
            # including train_data_source, train_result_target, predict_data_source

            '''
            for model in self.lib.models :
                if model.model_name == model_name :
                    model.get_config(config[model_name])
                    self.model_list.append(model)
                    break
            '''
            model = self.lib.models[model_name]
            model.get_config(arg_dict = config[model_name])
            self.model_list.append(model)
            # config 파일에 적힌 모델이 없는 경우에 대한 예외 처리 필요

        self.model_list[0].print_config_all(self.model_list)

    # read_training_data or predict_data
    # read_where can be 'db' or 'pipe' or 'queue' ...? 
    @abstractmethod
    def read_data(self, data_read_where):
        pass

    @abstractmethod
    def main(self):
        pass


