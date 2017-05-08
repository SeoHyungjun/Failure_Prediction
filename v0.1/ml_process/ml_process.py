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
import operation as op

from abc import abstractmethod

class ML_process_class :
    def __init__(self, config_fname='config'):
        self.model_num = 0
        self.model_dict = {}
        self.model_name_list = []
        self.cfg_name = config_fname
        self.predict_oper_list = [] # contain operation units
        self.train_oper_dict = {}   # key : first_model, second_model, .... value : operation unit list

    # config reads the config file and config
    # Configuration item will be what you use algorithm and data_transform...
    def config(self, config_fname='config'):
        config = cp.ConfigParser()
        config.read(config_fname)
        self.model_num = int(config['ML_Process']['model_num'])
        self.model_name_list = config['ML_Process']['model_names'] \
                                .replace(' ','').split(',')

        for section, entries in config.items() : # get model instance
            try :
                if section.split('_')[1] == 'MODEL' and entries['enable'] == 'true':
                    # print(items['model_name'])
                    model = library.class_obj_dict[entries['model_name']]()
                    model.set_config(arg_dict = entries)
                    self.model_dict[section.lower()] = model
                    # config 파일에 적힌 모델이 없는 경우에 대한 예외 처리 필요

            except IndexError:
                pass

        predict_operations_list = config['predict_operations']['predict_operations'] \
                                    .replace(' ', '').split(',')
        for oper in predict_operations_list:
            self.predict_oper_list.append(op.operation_unit(oper))

        train_operations_dict = config['train_operations'] # key : first_model value : D:"", T"", O"" ...
        for model_order, train_operations_str in train_operations_dict.items():
            train_operations_list = train_operations_str.replace(' ', '').split(',')
            self.train_oper_dict[model_order] = []
            for oper in train_operations_list:
                self.train_oper_dict[model_order].append(op.operation_unit(oper))

        self.print_config_all()

    def print_config_all(self):
        print("------------------------------------")
        print("Configuration information")
        print("------------------------------------")

        print("------------------------------------")
        self.model_dict['first_model'].print_config_all(self.model_dict)
        print("------------------------------------")

        print("------------------------------------")
        print("Prediction Operation Orders")
        for oper_unit in self.predict_oper_list:
            oper_unit.print_oper_unit()
        print("------------------------------------")

        print("------------------------------------")
        print("Training Operation Orders")
        for model_order, train_operations_list in self.train_oper_dict.items():
            print("%s" % model_order)
            for oper_unit in train_operations_list:
                oper_unit.print_oper_unit()
            print("------------------------------------")
        print("------------------------------------")

    # read_training_data or predict_data
    # read_where can be 'db' or 'pipe' or 'queue' ...? 
    @abstractmethod
    def read_data(self, data_read_where):
        pass

    @abstractmethod
    def main(self):
        pass
