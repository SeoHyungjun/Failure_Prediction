from model import *
import tensorflow as tf
import numpy as np
import make_input
import set_output_dir
import constant as ct

class CNN(Model):
    ### CV parameter ###

    def __init__(self, session, model_name="CNN", save_tag="model"):
        # make output directory
        self.saver_path, self.summary_train_path, self.summary_dev_path = set_output_dir.make_dir(model_name)
        # set output directory of tensorflow output
        self.model_prefix = os.path.join(self.saver_path, save_tag) 
        self.session = session

    def set_config(self):
        pass

    def create_model(self, x_height, x_width, num_NN_nodes, num_y_type, filter_sizes, num_filters, dropout_keep_prob=1.0, l2_reg_lambda=0.0):
    # x_height : height of input matrix
    # num_NN_nodes : fully connected NN nodes(array)  e.g. [3,4,5,2]
    # num_y_type : the number of output nodes. if regression, num_y_type == 1.
    # filter_sizes : list of size of filter matrix(two-dimention), [height, width]  e.g. [[1,2], [2,3], ...]
    # numb_filters : the number of each size of filter
    # regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)

    # pooling_size, dropout(Conv, NN), activation func, variable initializer

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, x_height, x_width], name="input_x")
        self.expanded_input_x = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(tf.int32, [None, num_y_type], name="input_y" )
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
       
        # Keeping track of 12 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        pooled_outputs = []

        # Convolution & Maxpooling layer
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size[0])):
                # Convolution Later
                filter_shape = [filter_size[0], filter_size[1], 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        
                conv = tf.nn.conv2d(
                    self.expanded_input_x,
                    W,                  # filter
                    strides=[1,1,1,1],
                    padding="VALID",    # no padding
                    name="conv")
                # Apply nonlinearity
                conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
       
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    conv_relu,
                    ksize=[1, x_height - filter_size[0] + 1, x_width - filter_size[1] + 1, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
        
        # Combine all the pooled featrues
        num_filters_total = num_filters * len(filter_sizes)
        pooled_concat = tf.concat(pooled_outputs, 3)
        pooled_flat = tf.reshape(pooled_concat, [-1, num_filters_total])

        with tf.name_scope("conv-dropout"):
            conv_drop = tf.nn.dropout(pooled_flat, dropout_keep_prob)
            
        # Hidden_NN layer
        pre_num_node = num_filters_total
        NN_result = [None] * (len(num_NN_nodes) + 1)
        NN_result[0] = conv_drop
        for index, num_node in enumerate(num_NN_nodes):
            if num_node == 0:
                index = -1
                break
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
                    NN_result[index+1] = tf.nn.dropout(NN_result[index+1], dropout_keep_prob)
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
            self.objective = tf.add(tf.reduce_mean(losses), (l2_reg_lambda * l2_loss), name="objective")
            tf.summary.scalar("loss", self.objective)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar("accuracy", accuracy)

        # Training operation
        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.objective)
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name="train_op")

        # Initialize all variables of tensor
        self.session.run(tf.global_variables_initializer())


    def restore_all(self, model_name="CNN", dir_root=ct.DIR_ROOT, graph_dir=ct.DIR_MODEL):
        checkpoint_file_path = os.path.join(dir_root, model_name, graph_dir)
        # Restore graph and variables and operation
        latest_model = tf.train.latest_checkpoint(checkpoint_file_path)
        restorer = tf.train.import_meta_graph("{}.meta".format(latest_model))
        restorer.restore(self.session, "{}".format(latest_model))
        # input operation
        self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
        self.input_y = self.session.graph.get_operation_by_name("input_y").outputs[0]
        self.dropout_keep_prob = self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # train operation
        self.train_op = self.session.graph.get_operation_by_name("train/train_op").outputs[0]
        self.global_step = self.session.graph.get_operation_by_name("train/global_step").outputs[0]
         

    def train(self, x, y, dev_sample_percentage, batch_size, num_epochs, evaluate_interval, save_interval, dropout_keep_prob=0.5):
    ### train parameter ###
    # dev_sample_percentage : percentage of the training data to use for validation"
    # batch_size : Batch Size
    # num_epochs : Number of training epochs
    # evaluate_interval : Evaluate model on dev set after this many steps (default: 150)
    # save_interval : Save model after this many steps (default: 150)
    # allow_soft_placement : Allow device soft device placement
    # log_device_placement : Log placement of ops on devices
    
        # make training/validation data batch by batch
        x_train, x_val, y_train, y_val = make_input.divide_fold(x, y, num_fold=10)
        batches = make_input.batch_iter(x_train, y_train, batch_size, num_epochs)

        # Summary writer(tensorboard) & Saver(save learned model), maybe not registered in graph
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.summary_train_path, self.session.graph)
        dev_writer = tf.summary.FileWriter(self.summary_dev_path, self.session.graph)
        model_saver = tf.train.Saver(tf.global_variables())

        # Training
        for batch in batches:
            x_batch = batch[0]
            y_batch = batch[1]
            feed_dict = {
                self.input_x : x_batch,
                self.input_y : y_batch,
                self.dropout_keep_prob : dropout_keep_prob
            }
            _, current_step, summary_train = self.session.run(
                [self.train_op, self.global_step, summary_op], feed_dict)
            train_writer.add_summary(summary_train, current_step)
            if current_step % save_interval == 0:
                model_saver.save(self.session, self.model_prefix, global_step=current_step)
                print("Save learned at step {}".format(current_step))
            if current_step % evaluate_interval == 0:
                feed_dict = {
                    self.input_x : x_val,
                    self.input_y : y_val,
                    self.dropout_keep_prob : 1.0
                }
                current_step, summary_dev = self.session.run(
                    [self.global_step, summary_op], feed_dict)
                print ("Eval model trained at step {}".format(current_step))
                dev_writer.add_summary(summary_dev, current_step)

    def run(self, x, y):
        feed_dict = {
            self.input_x : x,
            self.input_y : y,
            self.dropout_keep_prob : 1.0
        }
        result_op = self.session.graph.get_operation_by_name("output_layer/predictions").outputs[0]
        result = self.session.run([result_op], feed_dict)
        print (result)

