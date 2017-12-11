import pandas as pd
import numpy as np
import sys


class Data_transform:
    def __init__(self):
        pass
    
    def read_csv(self, data_path):
        return pd.read_csv(data_path)

    def sel_rows_by_index(self, data, start_index, end_index):
        if end_index > len(data) - 1:
            print("Error : end_index is larger than last_index")
            sys.exit()
        elif end_index is len(data) -1:
            return data.iloc[start_index:]
        else:
            return data.iloc[start_index:end_index+1]

    def sel_cols_by_index(self, data, cols_index):
        """
        input : cols_index is integer or list of integer.
        output : dataframe
        """
        return data.iloc[:,cols_index]

    def drop_cols_by_index(self, data, cols_index):
        """
        input : cols_index is integer or list of integer.
        output : dataframe
        """
        return data.drop(data.columns[cols_index], axis=1)

    def sel_cols_by_name(self, data, cols_name):
        """
        input : cols_name is string or list of string.
        output : dataframe
        """
        return data[cols_name]

    def drop_cols_by_name(self, data, cols_name):
        """
        input : cols_name is string or list of string.
        output : dataframe
        """
        return data.drop(cols_name, axis=1)

    def split_xy_by_yindex(self, data, y_cols=-1):
        """
        if y_cols is -1, y is last column
        output : dataframe
        """
        x = self.drop_cols_by_index(data, y_cols)
        y = self.sel_cols_by_index(data, y_cols)
        return x,y

    def split_xy_by_yname(self, data, y_cols):
        """
        output : dataframe
        """
        x = self.drop_cols_by_name(data, y_cols)
        y = self.sel_cols_by_name(data, y_cols)
        return x,y

    def create_window_data(self, data, y_cols, window_size, lead_time, strides=1):
        """
        1. split data for x and y. 
        2. make x, y data set as window-size and lead_time
        output : nparray
        """
        # split x, y as y_cols input type
        if isinstance(y_cols, list):
            if isinstance(y_cols[0], int):
                orig_x, orig_y = self.split_xy_by_yindex(data, y_cols)
            elif isinstance(y_cols[0], str):
                orig_x, orig_y = self.split_xy_by_yname(data, y_cols)
        elif isinstance(y_cols, int):
            orig_x, orig_y = self.split_xy_by_yindex(data, y_cols)
        elif isinstance(y_cols, str):
            orig_x, orig_y = self.split_xy_by_yname(data, y_cols)
        orig_x = orig_x.as_matrix()
        orig_y = orig_y.as_matrix()
         
        if strides is 0:
            strids = 1
        # make y
        index_first_y = window_size + lead_time - 1
        y = [label for i, label in enumerate(orig_y)
                if i - index_first_y >= 0 and ((i - index_first_y) % strides) == 0]
        y = np.array(y)
        # make x
        if window_size is 1:
            return orig_x, y
        x = []
        for i in range(len(y)):
            start_index = i * strides
            end_index = start_index + window_size
            x.append(orig_x[start_index:end_index])
        x = np.array(x)
        return x, y

    def _make_node_y_input(self, y_bundle, num_y_type):
        """
        ex) y is 'a' or 'b' => y will be [1,0] or [0,1]
        output : nparray
        """
        node_y_input = []
        for y in y_bundle:
            y_tmp = [0]*(num_y_type)
            y_tmp[int(y)] = 1
            node_y_input.append(y_tmp)
        node_y_input = np.array(node_y_input)
        return node_y_input
    
    def _divide_fold(self, x, y, num_fold):
        """
        randomly divide data into N fold. one fold is validation set.
        others are train set.
        """
        y_len = len(y)
        if y_len < 2:
            print("Too small data set. Need more data... exit.")
            sys.exit()
        # generate random list of numbers
        np.random.seed()
        shuffle_indices = np.random.permutation(np.arange(y_len))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]       
        # get boundary of train/dev
        if num_fold <= 0:
            print("Number of fold should be larger than 0")
            sys.exit()
        elif num_fold == 1:    # use all data set as training set
            val_percentage = 0 
        else:
            val_percentage = float(1) /float(num_fold)
        val_index = -1*int(val_percentage * float(y_len))
        if val_index == 0:
            val_index = -1   # num_fold == 1
        x_train, x_val = x_shuffled[:val_index], x_shuffled[val_index:]
        y_train, y_val = y_shuffled[:val_index], y_shuffled[val_index:]
        return x_train, x_val, y_train, y_val

    def _batch_iterator(self, x, y, batch_size, num_epochs):
        num_x = len(x)
        if num_x != len(y):
            print("the number of x and the number of y are different")
            sys.exit()
        # get the number of batches per epoch
        if num_x % batch_size == 0:
            num_batches_per_epoch = int(num_x/batch_size)
        else:
            num_batches_per_epoch = int(num_x/batch_size) + 1
        for epoch in range(num_epochs):
#            print("Start \'{}\' epoch\n".format(epoch + 1))
            for batch_index_per_epoch in range(num_batches_per_epoch):
                start_index = batch_index_per_epoch * batch_size
                end_index = min(start_index + batch_size, num_x)
                if end_index is num_x:
                    yield [x[start_index:], y[start_index:]]
                else :
                    yield [x[start_index:end_index], y[start_index:end_index]]


"""
data = pd.read_csv("./sample.csv")

dt = Data_transform()
x, y = dt._create_window_data(data,'failure', window_size=3,lead_time=0,strides=1)
x_train, x_val, y_train, y_val = dt._divide_fold(x,y,2)
batchs = dt._batch_generator(x_train, y_train, 3, 1)
for batch in batchs:
    print(batch)
    break
node_y_input = dt._make_node_y_input(y, 2)

data = dt.sel_cols_by_index(data, [0,1,2])
data = dt.sel_cols_by_name(data, ['failure', '1_normalized'])
data = dt.sel_rows_by_index(data, 0, 33)
"""
