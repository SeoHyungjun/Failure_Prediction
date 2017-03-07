#!/usr/bin/python3

import sys
sys.path.insert(0, '..')
import ml_process

class trainer(ml_process.ml_process_main) :
	def __init__(self):
		super.__init__()
		config()

	def config(self, config_fname='config'):
		super().config()

		config_parse(config_fname)

	def config_parse(self, config_fname):
		f=open(config_fname, 'r')

#		.... something parsing

		self.algo_num=1
		self.model_list.append('cnn')
		self.transform_list.append('transpose')
		self.data_read_where='file' # or 'db' or 'pipe' or 'msg q'

	def read_data(self, data_read_where):
		pass

	def main(self):
		self.read_data(


