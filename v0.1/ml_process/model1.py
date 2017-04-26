import os
from model import Model

class model1(Model):
    def __init__(self):
        self.model_name = "model1"
        self.arg_dict = {}
        self.param1 = 0
        self.param2 = ""
        self.param3 = ""
        self.training_data_source = ""
        self.train_operations = []

    def get_config(self, arg_dict) :
        self.arg_dict = arg_dict
        self.param1 = arg_dict['param1']
        self.param2 = arg_dict['param2']
        self.param3 = arg_dict['param3']
        self.training_data_source = arg_dict['training_data_source']
        self.train_operations = arg_dict['train_operations']

    def create_model(self):
        pass

    def restore_all(self):
        pass

    def train(self):
        pass

    def run(self):
        pass


