#!/usr/bin/python3

import ml_process

MODEL_ORDER = ['first_model', 'second_model', 'third_model', 'fourth_model']
# train_oper_dict is dictionary
# dict cannot iterate by order. So we use MODEL_ORDER list.

class Trainer(ml_process.ML_Process):
    def __init__(self):
        super().__init__()

    def main(self):
        self.config()

    '''
        for model_order in MODEL_ORDER:
            operations = self.train_oper_dict[model_order]
            model = self.model_dict[model_order]

            for operation_unit in operations:
                oper_type = operation_unit.oper_type
                func = operation_unit.execute_oper_func
                elif oper_type == 'DT':
                    pass
                elif oper_type == 'DP':
                    pass
                elif oper_type == 'OT':
                    pass
                elif oper_type == 'OP':
                    pass
                elif oper_type == 'R' or oper_type == 'T':
                    func(model, session_conf)
                elif oper_type == 'M':
                    pass
                else:
                    print("Operation Type is Wrong!!!. Type %s" % oper_type)
    '''
            #graph = tf.Graph()

        #for i -> n
        #self.model_list[i].train_operations




if __name__ == '__main__' :
    train = Trainer()
    train.main()
