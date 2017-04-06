import numpy as np
import sys
import pandas as pd


def split_xy(csv_file_path, x_height):
    data = pd.read_csv(csv_file_path)
#    data = open(csv_file_path, "r").readlines()

    x_maxlen = 0

    xy0 = next(data.iterrows())[1]
    xy0 = xy0.as_matrix()
    x_len = len(xy0[:-1])

    x = np.empty((0, x_height, x_len), int)
    window_queue = np.empty((x_height, x_len), int)
    y = np.empty((0, 1), int)

    for i, xy in data.iterrows():
        xy = xy.as_matrix()

        window_queue = np.delete(window_queue, 0, axis=0)
        window_queue = np.append(window_queue, [xy[:-1]], axis=0)
        
        if i+1 >= x_height:
            x = np.append(x, [window_queue], axis=0)
            y = np.append(y, xy[-1])

    if x_len == 0:
        print("x_len = {}. Input data parameter is deficient.".format(x_len))
        sys.exit()
    return x, x_len, y


def divide_fold(x, y, num_fold):
    """
    divide as N fold, select only one fold as validation set.
    other folds are used as training set
    """
    len_y = len(y)
    if len_y < 2:
        print("Too small data set. Need more data... exit.")
        sys.exit()

    # Randomly shuffle data
    np.random.seed()
    shuffle_indices = np.random.permutation(np.arange(len_y))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    # Split train/test set
    if num_fold <= 0:
        print("Number of fold should be larger than 0")
        sys.exit()
    elif num_fold == 1:    # use all data set as training set
        val_percentage = 0
    else:
        val_percentage = float(1) /float(num_fold)

    val_index = -1*int(val_percentage * float(len_y))
    if val_index == 0:
        val_index = -1   # num_fold == 1
    x_train, x_val = x_shuffled[:val_index], x_shuffled[val_index:]
    y_train, y_val = y_shuffled[:val_index], y_shuffled[val_index:]

    return x_train, x_val, y_train, y_val


def batch_iter(x, y, batch_size, num_epochs):
    x_size = len(x)
    if x_size != len(y):
        print("size of x and size of y are different")
        sys.exit()

    if x_size % batch_size == 0:
        num_batches_per_epoch = int(x_size/batch_size)
    else:
        num_batches_per_epoch = int(x_size/batch_size) + 1
    for epoch in range(num_epochs):
        for num_batch in range(num_batches_per_epoch):
            start_index = num_batch * batch_size
            end_index = min(start_index + batch_size, x_size)
            print("Start \'{}\' epoch\n".format(epoch + 1))
            yield [x[start_index:end_index], y[start_index:end_index]]
