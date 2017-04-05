from model import *
import tensorflow as tf
import numpy as np
import make_input
import set_out_dir
import constant as ct

class CNN(Model):
    ### CV parameter ###

    def __init__(self, directory):
    ### env parameter(init) ###
        pass


    def create_model(self, input_size, num_NN_nodes, num_output, filter_sizes, num_filters, dropout_keep_prob=1.0, l2_reg_lambda=0.0):
    # Model parameter of create_model
    # input_size : size of input matrix(two-dimention), [height, width]  e.g. [3,4]
    # num_NN_nodes : fully connected NN nodes(array)  e.g. [3,4,5,2]
    # num_output : the number of output nodes. if regression, num_output = 1.
    # filter_sizes : list of size of filter matrix(two-dimention)  e.g. [[1,2], [2,3], ...]
    # numb_filters : the number of each size of filter
    # regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)

    # pooling_size, dropout(Conv, NN), activation func, variable initializer

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1]], name="input_x")
        self.expanded_input_x = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(tf.int32, [None, num_output], name="input_y" )
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
       
        # Keeping track of 12 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        pooled_outputs = []
    
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
                    ksize=[1, input_size[0] - filter_size[0] + 1, input_size[1] - filter_size[1] + 1, 1],
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


        with tf.name_scope("output_layer"):
            W = tf.get_variable(
                "W",
                shape=[pre_num_node, num_output],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_output]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(NN_result[index+1], W, b, name="output")
            self.softmax = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + (l2_reg_lambda * l2_loss)

        with tf.name_scope("eval_info"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def _restore(self):
        pass
    
    def _save(self):
        print ("Save model trained!!!")
        pass
      
    def _eval(self):
        print ("Eval model trained!!!")
        pass


    def train(self, dev_sample_percentage, data_file_path, tag, batch_size, num_epochs, evaluate_every, saver_every, dropout_keep_prob=0.5, out_subdir=ct.STR_DERECTORY_ROOT):
    ### train parameter ###
    # dev_sample_percentage : percentage of the training data to use for validation"
    # data_file_path : Data source for training
    # tag : added in output directory name
    # batch_size : Batch Size
    # num_epochs : Number of training epochs
    # evaluate_every : Evaluate model on dev set after this many steps (default: 150)
    # saver_every : Save model after this many steps (default: 150)
    # allow_soft_placement : Allow device soft device placement
    # log_device_placement : Log placement of ops on devices
    # out_subdir : directory for saving output
    
        # make output directory
        set_out_dir.make_dir("CNN")

        # Load training/validation data batch by batch
        x, y = make_input.split_xy(csv_file_path=data_file_path)
        x_train, x_val, y_train, y_val = make_input.divide_fold(x, y, num_fold=10)
        batches = make_input.batch_iter(x_train, y_train, batch_size, num_epochs)

        # setting session for training 
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)


        # Training
        with tf.Session(config=session_conf) as sess:
            # 1. Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # 2. setting summary(tensorboard) and saver(save learned graph) operation
            loss_summary = tf.summary.scalar("loss", self.loss)
            acuracy_summary = tf.summary.scalar("acuracy", self.acuracy)
            summary_op = tf.summary.merge([loss.summary, acuracy_summary])

            summary_dir = os.paht.join(
            summary_writer = tf.train.SummaryWriter(out_subdir

            saver = tf.train.Saver(tf.all_variables())
             

            # 3. do training
            sess.run(tf.global_variables_initializer())
            for batch in batches:
                x_batch = batch[0]
                y_batch = batch[1]
                feed_dict = {
                    self.input_x : x_batch,
                    self.input_y : y_batch,
                    self.dropout_keep_prob : dropout_keep_prob
                }

                _, step, summary, loss, accuracy = sess.run(
                    [train_op, global_step, summary_op, self.loss, self.accuracy], feed_dict)
                
                

            """    
            print("End \'{}\' epoch\n".format(epoch + 1))
            _, run_result = sess.run([train_op, self.accuracy], feed_dict={self.input_x:[[[3,2],[5,4]], [[5,6],[8,9]]], self.input_y:[[0,1,0],[1,0,0]]})


        print(run_result)
        """
  
    def run(self):
        print ("Predict Something!!!!")
        pass

