#!/usr/bin/python3

import tensorflow as tf
import pandas as pd
import ml_process

class Predictor(ml_process.ML_process_class) :
    def __init__(self):
        super().__init__()
        self.predict_data_source = None # 나중에 디폴트 설정넣어놔야함

    def read_data(self, data_read_where):
        pass

    def main(self):
        self.config()

        session_conf = tf.ConfigProto(allow_soft_placement=True, \
                                     log_device_placement=False)

        for operation_unit in self.predict_oper_list:
            oper_type = operation_unit.oper_type
            func = operation_unit.execute_oper_func
            if oper_type == 'T':
                print("Operation type Train is observed in predictor")
                print("Check the operation order configuration")
                exit(1)

            if oper_type == 'D':
                self.predict_data_source = func()
            if oper_type == 'R':

                func(model, session_conf)
            if oper_type == 'O':
                pass
            if oper_type == 'M':
                pass


            #graph = tf.Graph()


        #for i -> n
        #self.model_list[i].train_operations




if __name__ == '__main__' :
    predictor = Predictor()
    predictor.main()
