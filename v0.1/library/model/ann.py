import sys
import tensorflow as tf
import numpy as np

from model import *
import make_input
import set_output_dir
import constant as ct

class ANN(Model):

<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, session, model_name="ANN", save_tag=ct.STR_SAVED_MODEL_PREFIX):
        # make output directory
        self.saver_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(model_name)
        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.saver_path, save_tag)
=======
    def __init__(self, session):
>>>>>>> 8662af41ade9c30a5a416dae4f2fd7ade6274dc8
        self.session = session
=======
    def __init__(self):
>>>>>>> 2ad6813a3f724d0197c5fc84639202a4741b1e43
        self.model_name = ct.ANN_MODEL_NAME
        self.model_dir = ct.ANN_MODEL_DIR
        # output config
        self.model_save_tag = ct.MODEL_SAVE_TAG
        self.project_dirpath = ct.PROJECT_DIRPATH
        # create_model
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.nodes_num = ct.ANN_NODES_NUM
        self.dropout_keep_prob = ct.ANN_DROPOUT_KEEP_PROB
        self.l2_reg_lambda = ct.ANN_L2_REG_LAMBDA
        self.validation_sample_percentage = ct.ANN_VALIDATION_SAMPLE_PERCENTAGE
        self.batch_size = ct.ANN_BATCH_SIZE
        self.epochs_num = ct.ANN_EPOCHS_NUM
        self.validation_interval = ct.ANN_VALIDATION_INTERVAL

    def create_model(self):
        self.model_dir = str(self.model_sequence) + '.' + self.model_dir
        # make output directory
        self.dirpath_trained_model, self.dirpath_summary_train, self.dirpath_summary_validation = set_output_dir.make_dir(self.model_dir, self.project_dirpath)
        self.model_filepath = os.path.join(self.dirpath_trained_model, self.model_save_tag)

<<<<<<< HEAD
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, x_width], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_y_type], name="input_y" )
<<<<<<< HEAD
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

=======
        # used when evaluation(keep_prob = 1.0)
        self.input_dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
       
>>>>>>> 8662af41ade9c30a5a416dae4f2fd7ade6274dc8
        # Keeping track of 12 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # ANN layer
        pre_num_node = x_width
        NN_result = [None] * (len(self.num_nodes) + 1)
        NN_result[0] = self.input_x
        for index, num_node in enumerate(self.num_nodes):
            if num_node == 0:
                print("the number of ANN layer node(num_node=0) is not valid")
                index = -1
                sys.exit()
            with tf.name_scope("completely_connected_NN_layer{}".format(index+1)):
=======
        x_width = self.x.shape[-1]
        num_y_type = self.y.shape[-1]
        with self.graph.as_default():
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.float32, [None, x_width], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None, num_y_type], name="input_y" )
            # used when evaluation(keep_prob = 1.0)
            self.input_dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
           
            # Keeping track of 12 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # ANN layer
            pre_num_node = x_width
            NN_result = [None] * (len(self.nodes_num) + 1)
            NN_result[0] = self.input_x
            for index, num_node in enumerate(self.nodes_num):
                if num_node == 0:
                    print("the number of ANN layer node(num_node=0) is not valid")
                    index = -1
                    sys.exit()
                with tf.name_scope("completely_connected_NN_layer{}".format(index+1)):
                    W = tf.get_variable(
                        "W_layer{}".format(index+1),
                        shape = [pre_num_node, num_node],
                        initializer = tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[num_node]), name = "b")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    NN_result[index+1] = tf.sigmoid(tf.nn.xw_plus_b(NN_result[index], W, b, name="NN_result{}".format(index+1)))
                    with tf.name_scope("dropout"):
                        NN_result[index+1] = tf.nn.dropout(NN_result[index+1], self.input_dropout_keep_prob)
                    pre_num_node = num_node
    
            # Predict & Classify layer
            with tf.name_scope("output_layer"):
>>>>>>> 2ad6813a3f724d0197c5fc84639202a4741b1e43
                W = tf.get_variable(
                    "W",
                    shape=[pre_num_node, num_y_type],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_y_type]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
<<<<<<< HEAD
                NN_result[index+1] = tf.sigmoid(tf.nn.xw_plus_b(NN_result[index], W, b, name="NN_result{}".format(index+1)))
                with tf.name_scope("dropout"):
                    NN_result[index+1] = tf.nn.dropout(NN_result[index+1], self.input_dropout_keep_prob)
                pre_num_node = num_node

        # Predict & Classify layer
        with tf.name_scope("output_layer"):
            W = tf.get_variable(
                "W",
                shape=[pre_num_node, num_y_type],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_y_type]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(NN_result[index+1], W, b, name="output")
            self.softmax = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Evaluation layer
        with tf.name_scope("eval_info"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.objective = tf.add(tf.reduce_mean(losses), (self.l2_reg_lambda * l2_loss), name="objective")
            tf.summary.scalar("loss", self.objective)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

        # Training operation
        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.objective)
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name="train_op")

        # Initiazlize all variables of tensor
        self.session.run(tf.global_variables_initializer())


    def restore_all(self):
        dirpath_model = os.path.join(self.dir_output, self.model_name)
        self.dirpath_trained_model = os.path.join(dirpath_model, ct.DIR_TRAINED_MODEL)
        self.dirpath_summary_train = os.path.join(dirpath_model, ct.DIR_SUMMARY, ct.DIR_SUMMARY_TRAIN)
        self.dirpath_summary_validation = os.path.join(dirpath_model, ct.DIR_SUMMARY, ct.DIR_SUMMARY_VALIDATION)
        checkpoint_file_path = os.path.join(self.dirpath_trained_model)
        # Restore graph and variables and operation
        latest_model = tf.train.latest_checkpoint(checkpoint_file_path)
        restorer = tf.train.import_meta_graph("{}.meta".format(latest_model))
        restorer.restore(self.session, "{}".format(latest_model))
        # input operation
        self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
        self.input_y = self.session.graph.get_operation_by_name("input_y").outputs[0]
        self.input_dropout_keep_prob = self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # train operation
        self.train_op = self.session.graph.get_operation_by_name("train/train_op").outputs[0]
        self.accuracy = self.session.graph.get_operation_by_name("eval_info/accuracy").outputs[0]
        self.global_step = self.session.graph.get_operation_by_name("train/global_step").outputs[0]
<<<<<<< HEAD


    def train(self, x, y, dev_sample_percentage, batch_size, num_epochs, evaluate_interval, save_interval, dropout_keep_prob=0.5):
    ### train parameter ###
    # dev_sample_percentage : percentage of the training data to use for validation"
    # batch_size : Batch Size
    # num_epochs : Number of training epochs
    # evaluate_interval : Evaluate model on dev set after this many steps (default: 150)
    # save_interval : Save model after this many steps (default: 150)
    # allow_soft_placement : Allow device soft device placement
    # log_device_placement : Log placement of ops on devices

=======
         
    def train(self, x, y):
>>>>>>> 8662af41ade9c30a5a416dae4f2fd7ade6274dc8
=======
    
                self.scores = tf.nn.xw_plus_b(NN_result[index+1], W, b, name="output")
                self.softmax = tf.nn.softmax(self.scores, name="softmax_scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
                
            # Evaluation layer
            with tf.name_scope("eval_info"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.objective = tf.add(tf.reduce_mean(losses), (self.l2_reg_lambda * l2_loss), name="objective")
                tf.summary.scalar("loss", self.objective)
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                tf.summary.scalar("accuracy", self.accuracy)
    
            # Training operation
            with tf.name_scope("train"):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.objective)
                self.op_train = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name="op_train")
    
            # Summary & Saver(save learned model)
            self.op_summary = tf.summary.merge_all()
            self.saver_model = tf.train.Saver(tf.global_variables(), name="saver_model")

            # Initiazlize all variables of tensor
            self.session.run(tf.global_variables_initializer())


    def restore_all(self):
        with self.graph.as_default():
            dirpath_model = os.path.join(self.project_dirpath, self.model_name)
            self.dirpath_trained_model = os.path.join(dirpath_model, ct.TRAINED_MODEL_DIR)
            self.dirpath_summary_train = os.path.join(dirpath_model, ct.SUMMARY_DIR, ct.SUMMARY_TRAIN_DIR)
            self.dirpath_summary_validation = os.path.join(dirpath_model, ct.SUMMARY_DIR, ct.SUMMARY_VALIDATION_DIR)
            checkpoint_file_path = os.path.join(self.dirpath_trained_model)
            # Restore graph and variables and operation
            latest_model = tf.train.latest_checkpoint(checkpoint_file_path)
            restorer = tf.train.import_meta_graph("{}.meta".format(latest_model))
            restorer.restore(self.session, "{}".format(latest_model))
            # input operation
            self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = self.session.graph.get_operation_by_name("input_y").outputs[0]
            self.input_dropout_keep_prob = self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # train operation
            self.op_train = self.session.graph.get_operation_by_name("train/op_train").outputs[0]
            self.accuracy = self.session.graph.get_operation_by_name("eval_info/accuracy").outputs[0]
            self.global_step = self.session.graph.get_operation_by_name("train/global_step").outputs[0]
            # summary, model saver
            self.op_summary = tf.summary.merge_all()
            self.saver_model = tf.train.Saver(tf.global_variables(), name="saver_model")
         
    def train(self):
>>>>>>> 2ad6813a3f724d0197c5fc84639202a4741b1e43
        # make training/validation data batch by batch
        x_train, x_val, y_train, y_val = make_input.divide_fold(self.x, self.y, num_fold=10)
        batches = make_input.batch_iter(x_train, y_train, self.batch_size, self.epochs_num)
        # writer(tensorboard) 
        writer_train = tf.summary.FileWriter(self.dirpath_trained_model, self.session.graph)
        writer_validation = tf.summary.FileWriter(self.dirpath_summary_train, self.session.graph)

        # Training
        for batch in batches:
            x_batch = batch[0]
            y_batch = batch[1]
            feed_dict = {
                self.input_x : x_batch,
                self.input_y : y_batch,
                self.input_dropout_keep_prob : self.dropout_keep_prob
            }
            _, current_step, summary_train = self.session.run(
                [self.op_train, self.global_step, self.op_summary], feed_dict)
            writer_train.add_summary(summary_train, current_step)
            if current_step % self.validation_interval == 0:
                feed_dict = {
                    self.input_x : x_val,
                    self.input_y : y_val,
                    self.input_dropout_keep_prob : 1.0
                }
                accuracy, summary_validation = self.session.run(
                    [self.accuracy, self.op_summary], feed_dict)
                print ("validation at step {}, accuracy = {}".format(current_step, accuracy))
                writer_validation.add_summary(summary_validation, current_step)
        # save trained model
        filepath_trained_model = os.path.join(self.dirpath_trained_model, self.model_save_tag) 
        self.saver_model.save(self.session, filepath_trained_model, global_step=current_step)
        print("Save learned at step {}".format(current_step))

    def run(self):
        feed_dict = {
            self.input_x : self.x,
            self.input_y : self.y,
            self.input_dropout_keep_prob : 1.0
        }
        op_result = self.session.graph.get_operation_by_name("output_layer/predictions").outputs[0]
        result = self.session.run([op_result], feed_dict)
        return result
