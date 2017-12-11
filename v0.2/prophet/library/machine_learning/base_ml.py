import sys
#sys.path.insert(0, '..')
import os
from abc import ABC, abstractmethod
#from eval_info import *

# abstract class to be used by machine learning class
class Machine_Learning(ABC):
    # db = Database()
    # input_x = Dataframe()
    # input_y = Dataframe()
    # ev_inf = Eval_info() 

    @abstractmethod
    def __init__(self):
        self.ml_name =''
        self.arg_dict = {}

    @abstractmethod
    def create_ml(self):
        pass
    
    @abstractmethod
    def restore_all(self):
        pass
  
    @abstractmethod
    def train(self):
        pass
  
    @abstractmethod
    def run(self):
        pass
    
    @staticmethod
    def print_config(ml_name, arg_dict):
        print("------------------------------------")
        print("Machine_Learning [%s] configuration information" % ml_name)
        for key in arg_dict.keys():
            print("[%s] : %s" % (key, arg_dict.get(key)))
        print("------------------------------------")

    @staticmethod
    def print_config_all(ml_list) :
        print("Configuration information")
        print("The Number of Machine_Learnings : %d" % len(ml_list))
        for ml in ml_list :
            ml.print_config(ml.ml_name, ml.arg_dict)

    @staticmethod
    def set_config(self, section_num, arg_dict):
        for config in arg_dict:
            apply_config_string = 'self.' + config + '=arg_dict[\'' + config + '\']'
            exec(apply_config_string)
        self.ml_sequence_num = section_num

    @staticmethod
    def set_x(self, x):
        self.x = x

    @staticmethod
    def set_y(self, y):
        self.y = y

#    @staticmethod
#    def set_ml_sequence(self, ml_sequence):
#        self.ml_sequence = ml_sequence
