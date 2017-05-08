#!/usr/bin/python3

import tensorflow as tf
import pandas as pd
import ml_process

MODEL_ORDER = ['first_model', 'second_model', 'third_model', 'fourth_model']
# train_oper_dict is dictionary
# dict cannot iterate by order. So we use MODEL_ORDER list.

class Trainer(ml_process.ML_process_class) :
    def __init__(self):
        super().__init__()
        self.train_data_source = None # 나중에 디폴트 설정넣어놔야함

    def read_data(self, data_read_where):
        pass

    def main(self):
        self.config()

        session_conf = tf.ConfigProto(allow_soft_placement=True, \
                                     log_device_placement=False)
    '''
        for model_order in MODEL_ORDER:
            operations = self.train_oper_dict[model_order]
            model = self.model_dict[model_order]

            for operation_unit in operations:
                oper_type = operation_unit.oper_type
                func = operation_unit.execute_oper_func
                if oper_type == 'D':
                    self.train_data_source = func()
                if oper_type == 'R' or oper_type == 'T':
                    func(model, session_conf)
                if oper_type == 'O':
                    pass
                if oper_type == 'M':
                    pass
    '''

            #graph = tf.Graph()


        #for i -> n
        #self.model_list[i].train_operations




if __name__ == '__main__' :
    train = Trainer()
    train.main()
