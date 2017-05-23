import sys
sys.path.insert(0, '..')
import os
from abc import ABC, abstractmethod
from eval_info import *

# abstract class to be used by machine learning class
class Model(ABC):
    # db = Database()
    # input_x = Dataframe()
    # input_y = Dataframe()
    ev_inf = Eval_info() 

    @abstractmethod
    def __init__(self):
        self.model_name =''
        self.arg_dict = {}

    @abstractmethod
    def create_model(self):
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
    def print_model_config(model_name, arg_dict):
        print("------------------------------------")
        print("Model [%s] configuration information" % model_name)
        for key in arg_dict.keys():
            print("[%s] : %s" % (key, arg_dict.get(key)))
        print("------------------------------------")

    @staticmethod
    def print_config_all(model_list) :
        print("Configuration information")
        print("The Number of Models : %d" % len(model_list))
        for model in model_list :
            model.print_model_config(model.model_name, model.arg_dict)

    @staticmethod
    def set_config(self, arg_dict):
        for config in arg_dict:
            apply_config_string = 'self.' + config + '=arg_dict[\'' + config + '\']'
            exec(apply_config_string)
  
    @staticmethod
    def set_x(self, x):
        self.x = x

    @staticmethod
    def set_y(self, y):
        self.y = y

    @staticmethod
    def set_model_sequence(self, model_sequence):
        self.model_sequence = model_sequence
