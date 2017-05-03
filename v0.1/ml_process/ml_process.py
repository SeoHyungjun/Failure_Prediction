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
import re

from abc import abstractmethod

class operation_unit :
    def __init__(self, oper_str):
        split_opers = oper_str.split(':')

        self.oper_type = split_opers[0]
        # just for initialize

        if self.oper_type == 'D':
            self.data_source = split_opers[1].split('"')[1]    # data source (for D type)
        elif self.oper_type == 'T':
            self.train_model = split_opers[1].split('"')[1]    # train model (for T type)
        elif self.oper_type == 'R':
            self.run_model = split_opers[1].split('"')[1]      # run model (for R type)
        elif self.oper_type == 'O':
            self.output_path = split_opers[1].split('"')[1]      # output path (for O type)
        elif self.oper_type == 'M':
            self.transform_func = split_opers[1].split('"')[1]    # data transform func (for M type)
            self.trsf_func_args = [arg.split('"')[1] for arg in split_opers[2:]]    # args for M type

    def print_oper_unit(self):
        print("[operation type] : %s" % self.oper_type)

        if self.oper_type == 'D':
            print("[Data source] : %s" % self.data_source)
        elif self.oper_type == 'T':
            print("[Train Model] : %s" % self.train_model)
        elif self.oper_type == 'R':
            print("[Run Model] : %s" % self.run_model)
        elif self.oper_type == 'O':
            print("[Output Path] : %s" % self.output_path)
        elif self.oper_type == 'M':
            print("[Func Name] : %s" % self.transform_func)
            i = 0
            for arg in self.trsf_func_args:
                i = i + 1
                print("[Argument %d] : %s " % (i,  arg))



class ML_process_class :
    def __init__(self, config_fname='config'):
        self.model_num = 0
        self.model_list = []
        self.model_name_list = []
        self.cfg_name = config_fname
        self.predict_order_list = [] # contain operation units
        self.train_order_dict = {}   # key : first_model, second_model, .... value : operation unit list

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
                    self.model_list.append(model)
                    # config 파일에 적힌 모델이 없는 경우에 대한 예외 처리 필요

            except IndexError:
                pass

        predict_operations_list = config['predict_operations']['predict_operations'] \
                                    .replace(' ', '').split(',')
        for oper in predict_operations_list:
            self.predict_order_list.append(operation_unit(oper))

        train_operations_dict = config['train_operations'] # key : first_model value : D:"", T"", O"" ...
        for model_order, train_operations_str in train_operations_dict.items():
            train_operations_list = train_operations_str.replace(' ', '').split(',')
            self.train_order_dict[model_order] = []
            for oper in train_operations_list:
                self.train_order_dict[model_order].append(operation_unit(oper))

        self.print_config_all()

    def print_config_all(self):
        print("------------------------------------")
        print("Configuration information")
        print("------------------------------------")

        print("------------------------------------")
        self.model_list[0].print_config_all(self.model_list)
        print("------------------------------------")

        print("------------------------------------")
        print("Prediction Operation Orders")
        for oper_unit in self.predict_order_list:
            oper_unit.print_oper_unit()
        print("------------------------------------")

        print("------------------------------------")
        print("Training Operation Orders")
        for model_order, train_operations_list in self.train_order_dict.items():
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
