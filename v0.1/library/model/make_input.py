import numpy as np
import sys

def split_xy(csv_file_path):
    data = open(csv_file_path, "r").readlines()

    xy0 = data[0]
    xy0 = xy0[:-1]
    xy0 = xy0.split(",")
    x = np.array([xy0[:-1]])
    y = np.array(xy0[-1])
    for xy in data[1:]:
        xy = xy[:-1]
        xy = xy.split(",")
        x = np.append(x, [xy[:-1]], axis=0)
        y = np.append(y, xy[-1])
    return x, y

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
