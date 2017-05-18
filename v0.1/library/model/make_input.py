import numpy as np
import sys
import pandas as pd


def split_xy(csv_file_path, num_y_type, x_height=1):
    """
    input csv file format = x1,x2,x3,x4,...,y
    'num_y_type' is range of y. ex) if y is 0 or 1, num_y_type==2.
    """
    data = pd.read_csv(csv_file_path)

    x_width = 0

    xy0 = next(data.iterrows())[1]
    xy0 = xy0.as_matrix()
    x_width = len(xy0[:-1])

    x = np.empty((0, x_height, x_width), int)
    x_window_queue = np.empty((x_height, x_width), int)
    y = np.empty((0, num_y_type), int)

    for i, xy in data.iterrows():
        xy = xy.as_matrix()

        x_window_queue = np.delete(x_window_queue, 0, axis=0)
        x_window_queue = np.append(x_window_queue, [xy[:-1]], axis=0)

        if i+1 >= x_height:
            x = np.append(x, [x_window_queue], axis=0)
            y_tmp = [0]*(num_y_type)
            y_tmp[int(xy[-1])] = 1
            y = np.append(y, [y_tmp], axis=0)

    if x_width == 0:
        print("x_width = {}. Input data parameter is deficient.".format(x_width))
        sys.exit()

    if x_height ==1:
        x = np.reshape(x, (-1, x_width))
    return x, x_width, y


def divide_fold(x, y, num_fold):
    """
    divide randomly as N fold, select only one fold as validation set.
    other folds are used as training set
    """
    num_y_type = len(y)
    if num_y_type < 2:
        print("Too small data set. Need more data... exit.")
        sys.exit()

    # Randomly shuffle data
    np.random.seed()
    shuffle_indices = np.random.permutation(np.arange(num_y_type))
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

    val_index = -1*int(val_percentage * float(num_y_type))
    if val_index == 0:
        val_index = -1   # num_fold == 1
    x_train, x_val = x_shuffled[:val_index], x_shuffled[val_index:]
    y_train, y_val = y_shuffled[:val_index], y_shuffled[val_index:]

    return x_train, x_val, y_train, y_val


def batch_iter(x, y, batch_size, num_epochs):
    num_x = len(x)
    if num_x != len(y):
        print("the number of x and the number of y are different")
        sys.exit()

    if num_x % batch_size == 0:
        num_batches_per_epoch = int(num_x/batch_size)
    else:
        num_batches_per_epoch = int(num_x/batch_size) + 1
    for epoch in range(num_epochs):
#        print("Start \'{}\' epoch\n".format(epoch + 1))
        for num_batch in range(num_batches_per_epoch):
            start_index = num_batch * batch_size
            end_index = min(start_index + batch_size, num_x)
            yield [x[start_index:end_index], y[start_index:end_index]]
