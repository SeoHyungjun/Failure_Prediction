import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from base_ml import Machine_Learning
FAILURE_PREDICTION_PATH = os.environ['FAILURE_PREDICTION']
#export FAILURE_PREDICTION=/root/Failure_Prediction-master/v0.2/prophet
sys.path.insert(0, os.path.join(FAILURE_PREDICTION_PATH, "library"))
from data_prepare import data_preprocessing as dp
import set_output_dir
import make_input
import constant as ct

class CNN(Machine_Learning):

	def __init__(self):
		self.graph = tf.Graph()
		self.session = tf.Session(graph=self.graph)

	def input(self):
		#self.x = pd.read_csv(self.train_inputpath)
		#self.y = self.x.iloc[:,-1].astype(int)
		#self.x = self.x.iloc[:,0:-1]
		#self.x = np.array(self.x)
		#self.y = dp.make_node_y_input(self.y, self.output_node_num)

		self.x, self.x_width, self.y = make_input.split_xy(csv_file_path="/root/FP_input/in_cnn.csv", num_y_type=self.num_y_type, x_height = self.x_height)
		print(self.x)
		print(self.x_width)
		print(self.y)

	def set_proper_config_type(self):
		print("set_proper_config_type")
		#read from config is string.
		#config에 적는 하이퍼파라미터 설정?!
		self.batch_size = int(self.batch_size)
		self.epochs_num = int(self.epochs_num)
		self.x_height = int(self.x_height)
		self.num_y_type = int(self.num_y_type)
		self.filter_sizes = self.filter_sizes.split("/")
		self.filter_sizes[0] = self.filter_sizes[0].split(",")
		self.filter_sizes[1] = self.filter_sizes[1].split(",")
		self.num_filters = int(self.num_filters)
		self.num_NN_nodes = [int(x) for x in self.num_nn_nodes.split(",")]
		self.l2_reg_lambda = float(self.l2_reg_lambda)
		self.save_interval = int(self.save_interval)
		self.evaluate_interval = int(self.evaluate_interval)

	def create_ml(self):
		print("create_ml")
		self.ml_dir = str(self.ml_sequence_num) + '_' + self.ml_dir

		self.dirpath_trained_model, self.dirpath_summary_train, self.dirpath_summary_validation = set_output_dir.make_dir(self.ml_dir, self.project_dirpath)
		self.model_filepath = os.path.join(self.dirpath_trained_model, self.trained_ml_save_tag)

		x_width = self.x.shape[-1]
		y_width = self.y.shape[-1]
		with self.graph.as_default():
			self.input_x = tf.placeholder(tf.float32, [None, self.x_height, x_width], name="input_x")
			self.expanded_input_x = tf.expand_dims(self.input_x, -1)
			self.input_y = tf.placeholder(tf.int32, [None, self.num_y_type], name="input_y" )
			self.input_dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 

			l2_loss = tf.constant(0.0)
			pooled_outputs = []

			for i, filter_size in enumerate(self.filter_sizes):
				print("1")
				with tf.name_scope("conv-maxpool-{}".format(filter_size[0])):
					print("2")
					filter_shape = [int(filter_size[0]), int(filter_size[1]), 1, self.num_filters]
					print(filter_shape)
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
					
					print("2.1")
					conv = tf.nn.conv2d(
						self.expanded_input_x,
						W,
						strides=[1,1,1,1],
						padding="VALID",
						name="conv")
					print("2.2")
					conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

					pooled = tf.nn.max_pool(
						conv_relu,
						ksize=[1, self.x_height - int(filter_size[0]) + 1, x_width - int(filter_size[1]) + 1, 1],
						strides=[1,1,1,1],
						padding="VALID",
						name="pool")
					pooled_outputs.append(pooled)
			print("2.3")
			num_filters_total = self.num_filters * len(self.filter_sizes)
			pooled_concat = tf.concat(pooled_outputs, 3)
			pooled_flat = tf.reshape(pooled_concat, [-1, num_filters_total])

			with tf.name_scope("conv-dropout"):
				print("3")
				conv_drop = tf.nn.dropout(pooled_flat, self.input_dropout_keep_prob)
			print("4")
			pre_num_node = num_filters_total
			NN_result = [None] * (len(self.num_NN_nodes) + 1)
			NN_result[0] = conv_drop
			for index, num_node in enumerate(self.num_NN_nodes):
				print("5")
				if num_node == 0:
					index = -1
					break
				with tf.name_scope("completely_connected_NN_layer{}".format(index+1)):
					print("6")
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

			with tf.name_scope("output_layer"):
				print("7")
				W = tf.get_variable(
					"w",
					shape=[pre_num_node, self.num_y_type],
					initializer=tf.contrib.layers.xavier_initializer())
				b = tf.Variable(tf.constant(0.1, shape=[self.num_y_type]), name="b")
				l2_loss += tf.nn.l2_loss(W)
				l2_loss += tf.nn.l2_loss(b)

				self.scores = tf.nn.xw_plus_b(NN_result[index+1], W, b, name="output")
				self.softmax = tf.nn.softmax(self.scores, name="softmax_scores")
				self.predictions = tf.argmax(self.scores, 1, name="predictions")

			with tf.name_scope("eval_info"):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
				self.objective = tf.add(tf.reduce_mean(losses), (self.l2_reg_lambda * l2_loss), name="objective")
				tf.summary.scalar("loss", self.objective)
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
				tf.summary.scalar("accuracy", self.accuracy)
	
			with tf.name_scope("train"):
				self.global_step = tf.Variable(0, name="global_step", trainable=False)
				optimizer = tf.train.AdamOptimizer(1e-3)
				grads_and_vars = optimizer.compute_gradients(self.objective)
				self.op_train = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name="op_train")

			self.op_summary = tf.summary.merge_all()
			self.saver_model = tf.train.Saver(tf.global_variables(), name="saver_model")			

			self.session.run(tf.global_variables_initializer())
			
		print("create end")
	
	def restore(self):
		print("restore")
		self.ml_dir = str(self.ml_sequence_num) + '_' + self.ml_dir
		dirpath_model = os.path.join(self.project_dirpath, self.ml_dir)
		self.dirpath_trained_model = os.path.join(dirpath_model, ct.TRAINED_MODEL_DIR)
		self.dirpath_summary_train = os.path.join(dirpath_model, ct.SUMMARY_DIR, ct.SUMMARY_TRAIN_DIR)
		self.dirpath_summary_validation = os.path.join(dirpath_model, ct.SUMMARY_DIR, ct.SUMMARY_VALIDATION_DIR)
		checkpoint_file_path = os.path.join(self.dirpath_trained_model)
		latest_model = tf.train.latest_checkpoint(checkpoint_file_path)
		with self.graph.as_default():
			restorer = tf.train.import_meta_graph("{}.meta".format(latest_model))
			restorer.restore(self.session, "{}".format(latest_model))
			self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
			self.input_y = self.session.graph.get_operation_by_name("input_y").outputs[0]
			self.input_dropout_keep_prob = self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
			self.op_train = self.session.graph.get_operation_by_name("train/op_train").outputs[0]
			self.accuracy = self.session.graph.get_operation_by_name("eval_info/accuracy").outputs[0]
			self.global_step = self.session.graph.get_operation_by_name("train/global_step").outputs[0]
			self.op_summary = tf.summary.merge_all()
			self.saver_model = tf.train.Saver(tf.global_variables(), name="saver_model")


	def train(self):
		print("train")
		x_train, x_val, y_train, y_val = make_input.divide_fold(self.x, self.y, num_fold=10)
		batches = make_input.batch_iter(x_train, y_train, self.batch_size, self.epochs_num)

		#self.op_summary = tf.summary.merge_all()
		#self.saver_model = tf.train.Saver(tf.global_variables(), name="saver_model")
		#train_writer = tf.summary.FileWriter(self.summary_train_path, self.session.graph)
		#dev_writer = tf.summary.FileWriter(self.summary_dev_path, self.session.graph)
		#model_saver = tf.train.Saver(tf.global_variables())

		writer_train = tf.summary.FileWriter(self.dirpath_trained_model, self.session.graph)
		writer_validation = tf.summary.FileWriter(self.dirpath_summary_train, self.session.graph)

		filepath_trained_model = os.path.join(self.dirpath_trained_model, self.trained_ml_save_tag)

		for batch in batches:
			#print(batch[0][0][0][0])
			x_batch = batch[0]
			y_batch = batch[1]
			feed_dict = {
				self.input_x : x_batch,
				self.input_y : y_batch,
				self.input_dropout_keep_prob : self.dropout_keep_prob
			}

			_, current_step, summary_train = self.session.run([self.op_train, self.global_step, self.op_summary], feed_dict)
			writer_train.add_summary(summary_train, current_step)
			if current_step % self.save_interval == 0:
				self.saver_model.save(self.session, filepath_trained_model, global_step=current_step)
				print("Save learned at step {}".format(current_step))
			
			if current_step % self.evaluate_interval == 0:
				feed_dict = {
					self.input_x : x_val,
					self.input_y : y_val,
					self.input_dropout_keep_prob : 1.0
				}
				accuracy, summary_validation = self.session.run(
					[self.accuracy, self.op_summary], feed_dict)
				print ("Eval model trained at step {}".format(current_step))
				writer_validation.add_summary(summary_validation, current_step)

		self.saver_model.save(self.session, filepath_trained_model, global_step=current_step)
		print("Save learned at step {}".format(current_step) + 'at ' + filepath_trained_model)


	def run(self):
		print("run")
		feed_dict = {
			self.input_x : self.x,
			self.input_y : self.y,
			self.input_dropout_keep_prob : 1.0
		}
		op_result = self.session.graph.get_operation_by_name("output_layer/predictions").outputs[0]
		result = self.session.run([op_result], feed_dict)
		output_filepath = os.path.join(self.project_dirpath, self.ml_dir, self.run_result_file)
		output = pd.DataFrame(data=result).T
		output.columns = ['result']
		output.to_csv(output_filepath, index=False)
		print("result saved as \'{}\'".format(output_filepath))
		return result
