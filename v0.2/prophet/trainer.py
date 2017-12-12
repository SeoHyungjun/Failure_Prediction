#!/usr/bin/python3

import ml_process

ML_ORDERS = ['1st_ml', '2nd_ml']
# train_oper_dict is dictionary
# dict cannot iterate by order. So we use MODEL_ORDER list.

class Trainer(ml_process.ML_Process):
    def __init__(self):
        super().__init__()

    def main(self):
        self.ml_instance_dict, self.train_oper_dict = self.get_train_configured_instance_operation()
        # execute operation in model order
        for ml_order in ML_ORDERS:
#        for ml_order, ml_instance in self.ml_instance_dict.items():
            ml_instance = self.ml_instance_dict[ml_order]
            # e.g. ml_order == 1st_ml
            operations = self.train_oper_dict[ml_order]  # operations have each ml's operation
            # each operation already have function to execute and parameter. only need to select what function to use. 
            for operation_unit in operations:
                oper_type = operation_unit.oper_type
                if oper_type == 'DP':
                    operation_unit.execute_oper_func()
                else:
                    operation_unit.execute_oper_func(ml_instance)
            
if __name__ == '__main__' :
    train = Trainer()
    train.main()
