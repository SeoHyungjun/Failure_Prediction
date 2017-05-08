#!/usr/bin/python3

import tensorflow as tf

class operation_unit :
    def __init__(self, oper_str):
        split_opers = oper_str.split(':')

        self.oper_type = split_opers[0]
        self.execute_oper_func = None
        # just for initialize

        if self.oper_type == 'D':
            self.data_source = split_opers[1].split('"')[1]    # data source (for D type)
            self.execute_oper_func = self.oper_D_type
        elif self.oper_type == 'T':
            self.train_model = split_opers[1].split('"')[1]    # train model (for T type)
            self.execute_oper_func = self.oper_T_type
        elif self.oper_type == 'R':
            self.run_model = split_opers[1].split('"')[1]      # run model (for R type)
            self.execute_oper_func = self.oper_R_type
        elif self.oper_type == 'O':
            self.output_path = split_opers[1].split('"')[1]      # output path (for O type)
            self.execute_oper_func = self.oper_O_type
        elif self.oper_type == 'M':
            self.transform_func = split_opers[1].split('"')[1]    # data transform func (for M type)
            self.trsf_func_args = [arg.split('"')[1] for arg in split_opers[2:]]    # args for M type
            self.execute_oper_func = self.oper_M_type

    def print_oper_unit(self):
        print("[operation type] : %s" % self.oper_type)

        if self.oper_type == 'D':
            print("[Data source] : %s" % self.data_source)
        elif self.oper_type == 'T':
            print("[Train Model] : %s" % self.train_model)
        elif self.oper_type == 'R':
            print("[Run Model] : %s" % self.run_model)
        elif self.oper_type == 'O':
            print("[Output Path] : %s" % self.output_path)
        elif self.oper_type == 'M':
            print("[Func Name] : %s" % self.transform_func)
            i = 0
            for arg in self.trsf_func_args:
                i = i + 1
                print("[Argument %d] : %s " % (i,  arg))

    # operation for D type. just return data source path.
    def oper_D_type(self):
        return self.data_source

    # operation for T type.
    def oper_T_type(self, model=None, session_conf=None, retrain=False):
        if model is None:
            print("Operation type is Train.")
            print("But, it doesn't exist model instance.!")
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
            model.set_session(sess)

            if not retrain:
                model.create_model()
            else:
                model.restore_all()

            model.train()

    def oper_R_type(self, model=None, session_conf=None):
        if model is None:
            print("Operation type is Run.")
            print("But, it doesn't exist model instance.!")
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
            model.set_session(sess)

            model.run()

    def oper_O_type(self):
        pass

    def oper_M_type(self):
        pass



