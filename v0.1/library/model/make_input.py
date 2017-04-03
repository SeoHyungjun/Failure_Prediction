import numpy as np
import sys

def make_N_fold(x, y, num_fold):
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
    else
        val_percentage = float(1) /float(num_fold)

    val_index = -1*int(val_percentage * float(len_y))
    if val_index == 0:
        val_index = -1   # num_fold == 1
    x_train, x_val = x_shuffled[:val_index], x_shuffled[val_index:]
    y_train, y_val = y_shuffled[:val_index], y_shuffled[val_index:]

    return x_train, x_val, y_train, y_val


def batch_iter(data, batch_size, num_epochs):
    data_size = len(data)
    if data_size % batch_size == 0:
        num_batches_per_epoch = int(data_size/batch_size)
    else:
        num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_batches_per_epoch):
        for batch_num in range(


