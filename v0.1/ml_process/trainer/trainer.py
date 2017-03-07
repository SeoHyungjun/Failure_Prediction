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

		if self.read_where == 'file':
			config_from_file()
		elif self.read_where == 'DB':
			config_from_db()

	def config_parse(self, config_fname):
		f=open(config_fname, 'r')

#		.... something parsing

		self.algo_num=1
		self.model_list.append('cnn')
		self.transform_list.append('transpose')
		self.read_where='file' # or 'db'
		pass

	def config_from_db(self):
		pass

	def config_from_file(self):
		pass

	def main(self):


