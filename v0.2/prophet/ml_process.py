#!/bin/python3
#-*- coding:utf-8 -*-

import os
import sys
FAILURE_PREDICTION_PATH = os.environ['FAILURE_PREDICTION']
sys.path.insert(0, FAILURE_PREDICTION_PATH) # upper directory
from library import get_classes
import configparser as cp
import operation as op

from abc import abstractmethod

g_config_filename = 'config'

class ML_Process :
    def __init__(self):
#        self.ml_num = 0
#        self.ml_name_list = []
#        self.ml_instance_dict = dict()
#        self.train_oper_dict = dict()   # key : first_ml, second_ml, .... value : operation unit list
#        self.predict_oper_dict = dict() # key : first_ml, second_ml, .... value : operation unit list
        pass

    # config reads the config file and config
    # Configuration item will be what you use. such as algorithm and data_transform...
    def get_train_instance_operation(self, cfg_fname=g_config_filename):
        ml_instance_dict = dict()
        ml_enable_list = []
        train_oper_dict = dict()   # key : first_ml, second_ml, .... value : operation unit list
        config = cp.ConfigParser()
        config.read(cfg_fname)
#        self.ml_num = int(config['ML_Process']['ml_num'])
#        self.ml_name_list = config['ML_Process']['ml_names'] \
#                                .replace(' ','').split(',')

        # every ml config section gets machine learning class instance
        for section, entries in config.items() : 
            try :
                if section.split('_')[1] == 'ML' and entries['enable'] == 'true':
                    ml_enable_list.append(section.lower())
                    ml_instance = get_classes.class_dict[entries['ML_NAME']]()      # get class instance
                    ml_instance.set_config(ml_instance, section_num = section[0], arg_dict = entries)         # set each config
                    ml_instance_dict[section.lower()] = ml_instance    # section.lower() = 1st_ML, 2nd_ML, ...
            # when the machine learning written in config file doesn't exist, exception process is needed.
            except IndexError:
                pass

        # get train_operations assing func as according to their type
        train_operations_dict = config['Train_Operations'] # key:first_ml ; value:I:"", T, O:"" ...
        for ml_order, train_operations_str in train_operations_dict.items():
            if ml_order in ml_enable_list:
                train_operations_list = train_operations_str.replace(' ', '').split(',')
                train_oper_dict[ml_order.lower()] = []
                for oper in train_operations_list:
                    train_oper_dict[ml_order.lower()].append(op.operation_unit(oper))   # set each operation and input path as operation_unit
        return (ml_instance_dict, train_oper_dict)

    # config reads the config file and config
    # Configuration item will be what you use. such as algorithm and data_transform...
    def get_run_instance_operation(self, cfg_fname=g_config_filename):
        ml_instance_dict = dict()
        ml_enable_list = []
        run_oper_dict = dict() # key : first_ml, second_ml, .... value : operation unit list 
        config = cp.ConfigParser()
        config.read(cfg_fname)
#        self.ml_num = int(config['ML_Process']['ml_num'])
#        self.ml_name_list = config['ML_Process']['ml_names'] \
#                                .replace(' ','').split(',')

        # every ml config section gets machine learning class instance
        for section, entries in config.items() : 
            try :
                if section.split('_')[1] == 'ML' and entries['enable'] == 'true':
                    # print(items['model_name'])
                    ml_instance = get_classes.class_dict[entries['ML_NAME']]()      # get class instance
                    ml_instance.set_config(ml_instance, section_num = section[0], arg_dict = entries)
                    ml_instance_dict[section.lower()] = ml_instance    # section.lower() = 1st_ml, 2nd_ml, ...
            # when the machine learning written in config file doesn't exist, exception process is needed.
            except IndexError:
                pass
        print(ml_enable_list)

        # get run_operations assing func as according to their type
        run_operations_dict = config['Predict_Operations'] # key:first_ml ; value:I:"", T, O:"" ...
        for ml_order, run_operations_str in run_operations_dict.items():
            if ml_order in ml_enable_list:
                run_operations_list = run_operations_str.replace(' ', '').split(',')
                run_oper_dict[ml_order.lower()] = []
                for oper in run_operations_list:
                    run_oper_dict[ml_order.lower()].append(op.operation_unit(oper))   # set each operation and input path as operation_unit
        return (ml_instance_dict, run_oper_dict)
        '''
        predict_operations_list = config['predict_operations']['predict_operations'] \
                                    .replace(' ', '').split(',')
        for oper in predict_operations_list:
            self.predict_oper_list.append(op.operation_unit(oper))

        train_operations_dict = config['Train_operations'] # key : first_model value : D:"", T"", O"" ...
        for model_order, train_operations_str in train_operations_dict.items():
            train_operations_list = train_operations_str.replace(' ', '').split(',')
            self.train_oper_dict[model_order] = []
            for oper in train_operations_list:
                self.train_oper_dict[model_order].append(op.operation_unit(oper))

        self.print_config_all()
        '''

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

    @abstractmethod
    def main(self):
        pass
