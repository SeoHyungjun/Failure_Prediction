import os
from model import Model

class model2(Model):
    def __init__(self):
        self.model_name = "model2"
        self.arg_dict = {}
        self.arg1 = 0
        self.arg2 = ""
        self.arg3 = ""
        self.training_data_source = ""
        self.train_operations = []

    def get_config(self, arg_dict):
        self.arg_dict = arg_dict
        self.arg1 = arg_dict['arg1']
        self.arg2 = arg_dict['arg2']
        self.arg3 = arg_dict['arg3']
        self.data_source = arg_dict['training_data_source']
        self.train_operations = arg_dict['train_operations']

    def create_model(self):
        pass

    def restore_all(self):
        pass

    def train(self):
        pass

    def run(self):
        pass



