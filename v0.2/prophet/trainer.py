#!/usr/bin/python3

import ml_process

class Trainer(ml_process.ML_Process):
    def __init__(self):
        super().__init__()

    def main(self):
        self.ml_instance_dict, self.train_oper_dict = self.get_train_instance_operation()
        
        for ml_order, ml_instance in self.ml_instance_dict.items():
            operations = self.train_oper_dict[ml_order]
            
            # each operation already have function to execute and parameter. only need to select what function to use. 
            for operation_unit in operations:
                oper_type = operation_unit.oper_type
                operation_unit.execute_oper_func(ml_instance)
            
            # graph = tf.Graph()
        
        #for i -> n
        #self.model_list[i].train_operations

if __name__ == '__main__' :
    train = Trainer()
    train.main()
