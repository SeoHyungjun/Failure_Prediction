#!/usr/bin/python3

import sys
sys.path.insert(0, '..')
import ml_process

class Trainer(ml_process.ML_process_class) :
    def __init__(self):
        super().__init__()

    def read_data(self, data_read_where):
        pass

    def main(self):
        self.config()


if __name__ == '__main__' :
    train = Trainer()
    train.main()
    train.print_config()
