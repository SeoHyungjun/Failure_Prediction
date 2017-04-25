#!/bin/python3

# import sys
# sys.path.insert(0, '../library/data_transform')
# sys.path.insert(0, '../library/model')
# import eval_info
# import cnn
# import library
import configparser as cp

from abc import ABCMeta
from abc import abstractmethod

class ML_process_class :
    def __init__(self, config_fname='config'):
        self.algo_num = 0
        self.model_list = []
        self.transform_dict = dict()
        self.algoparam_dict = dict()
        self.cfg_name = config_fname

    # config reads the config file and config
    # Configuration item will be what you use algorithm and data_transform...
    def config(self, config_fname='config'):
        config = cp.ConfigParser()
        config.read(config_fname)
        self.algo_num = int(config['ML_Process']['algo_num'])
        self.model_list = config['ML_Process']['algo_names'] \
                                .replace(' ','').split(',')

        for model in self.model_list :
            # transform_dict has the data transform functions that 
            # will work before the algorith is executed 
            self.transform_dict[model] = config[model]['data_transform_names'] \
                                            .replace(' ','').split(',')

            # algoparam_dict has each algorithm's parameters
            # including train_data_source, train_result_target, predict_data_source
            self.algoparam_dict[model] = dict()
            self.algoparam_dict[model]['train_data_source'] = \
                                    config[model]['train_data_source']
            self.algoparam_dict[model]['train_result_target'] = \
                                    config[model]['train_result_target']
            self.algoparam_dict[model]['predict_data_source']= \
                                    config[model]['predict_data_source']

            if model == 'CNN' :
                self.algoparam_dict[model]['nr_layer'] = \
                                    config[model]['NR_Layer']
            elif model == 'Kmeans' :
                self.algoparam_dict[model]['cluster_num'] = \
                                    config[model]['cluster_num']
            elif model == 'ANN' :
                self.algoparam_dict[model]['nr_layer'] = \
                                    config[model]['NR_Layer']

    def print_config(self) :
        print("Configuration information")
        print("{0:20s} : ".format("Config file path") + "%s" % self.cfg_name)
        print("{0:20s} : ".format("Number of Algorithm") + "%d" % self.algo_num)
        print("{0:20s} : ".format("Algorithms") + ' '.join(self.model_list))

        for model in self.model_list :
            print("\nModel [%s] Configuration" % model)
            param_keys = list(self.algoparam_dict[model].keys())

            for param in param_keys :
                print("{0:20s} : ".format(param) \
                            + "{}".format(self.algoparam_dict[model][param]))


            '''
            print("Train data source : ".ljust(20) + "%s",
                            self.algoparam_dict[model]['train_data_source'])
            print("Train result source : ".ljust(20) + "%s",
                            self.algoparam_dict[model]['train_result_target'])
            print("Predict data source : ".ljust(20) + "%s",
                            self.algoparam_dict[model]['predict_data_source'])
            '''

    # read_training_data or predict_data
    # read_where can be 'db' or 'pipe' or 'queue' ...? 
    @abstractmethod
    def read_data(self, data_read_where):
        pass

    @abstractmethod
    def main(self):
        pass


