#!/usr/bin/python3

import tensorflow as tf

class operation_unit :
    def __init__(self, oper_str):
        split_opers = oper_str.split(':')

        self.oper_type = split_opers[0]
        self.execute_oper_func = None

        if self.oper_type == 'create':
            self.execute_oper_func = self.oper_create_type
        elif self.oper_type == 'restore':
            self.execute_oper_fucn = self.oper_restore_type
        elif self.oper_type == 'train':
            self.execute_oper_func = self.oper_train_type
        elif self.oper_type == 'run':
            self.execute_oper_fucn = self.oper_run_type
#        elif self.oper_type == 'DP':
#            self.dt_func = split_opers[1].split('"')[1]
#            self.dt_func_args = [arg.split('"')[1] for arg in split_opers[2:]] # args for DT type
#            self.execute_oper_func = self.oper_DP_type

    def print_oper_unit(self):
        print("[operation type] : %s" % self.oper_type)
        '''
        if self.oper_type == 'I':
            print("[Train Input Path] : %s" % self.train_input_path)
        elif self.oper_type == 'PI':
            print("Predict Input Path] : %s" % self.predict_input_path)
        elif self.oper_type == 'O':
            print("[Train Output Path] : %s" % self.train_output_path)
        elif self.oper_type == 'PO':
            print("[Predict Output Path] : %s" % self.predict_output_path)
        elif self.oper_type == 'T':
            print("[Train Model] : %s" % self.train_model)
        elif self.oper_type == 'R':
            print("[Run Model] : %s" % self.run_model)
        elif self.oper_type == 'DT':
            print("[Func Name] : %s" % self.transform_func)
            i = 0
            for arg in self.trsf_func_args:
                i = i + 1
                print("[Argument %d] : %s " % (i,  arg))
        '''

    def oper_create_type(self, ml_instance):
        print("create {}".format(ml_instance.ml_name))
        ml_instance.create_ml()
        pass

    def oper_restore_type(self, ml_instance):
        print("###### restore ########")
        pass

    def oper_train_type(self, ml_instance):
        print("###### train ########")
#        ml_instance.train()
        pass

    def oper_run_type(self, ml_instance):
        print("###### run ########")
        pass

#    def oper_PO_type(self, model, path):
#        pass

    '''
    # operation for T type.
    def oper_T_type(self, ml_instace=None, retrain=False):
        if machine_learning is None:
            print("Operation type is Train.")
            print("But, it doesn't exist machine learning model instance.!")
            print("exit...")
            exit(1)

            if not retrain:
                ml_instance.create_model()
            else:
                ml_instance.restore_all()

            ml_instacne.train()

    def oper_R_type(self, ml_instance=None, session_conf=None):
        if ml_instacne is None:
            print("Operation type is Run.")
            print("But, machine_learning instance doesn't exist.!")
            print("exit...")
            exit(0)
        elif session_conf is None:
            print("Session configuration is not given")
            print("Setting on default session config")
            print("allow_soft_placement=True, log_device_placement=False")
            session_conf = tf.ConfigProto(allow_soft_placement=True, \
                                          log_device_placement=False)

        graph = tf.Graph()

        with tf.Session(graph=graph, config=session_conf) as sess:
            machine_learning.set_session(sess)

            machine_learning.run()

    def oper_DT_type(self):
        func_name = self.dt_func
        args = self.dt_func_args

        func = getattr(data_transform, func_name)

        ret = func(*args)

        if ret is not None:
            return ret
    '''
