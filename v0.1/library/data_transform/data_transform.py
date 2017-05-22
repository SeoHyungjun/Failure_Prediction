# data_transform.py

import pandas as pd
import numpy as np

class data_transform :
    def __init__(self):
        pass

    # data   : dataframe type
    # y_cols : y data column index list
    # return : split x, y data in tuple, dataframe type
    #
    def split_x_y(self, data, y_cols):
        x = pd.DataFrame()
        y = pd.DataFrame()
        all_cols = list(data)

        x_cols = [i for j, i in enumerate(all_cols) if j not in y_cols]
        y_cols = [i for j, i in enumerate(all_cols) if j in y_cols]

        for idx in y_cols:
            y[idx] = data[idx]

        for idx in x_cols:
            x[idx] = data[idx]

        return x, y

    # data          : origin data, dataframe type
    # y cols        : y data column index list
    # window_size   : window size
    # lead time     : distance between last window element and y value
    # strides       : interval between adjacent windows
    # return        : NULL
    #
    def create_window_data(self, data, y_cols, window_size, lead_time, strides=1):
        orig_x, orig_y = self.split_x_y(data, y_cols)
        orig_x = orig_x.as_matrix()
        orig_y = orig_y.as_matrix()

        if strides == 0: # strides could not be 0
            strides = 1

        idx_first_y = window_size + lead_time - 1
        print(idx_first_y)
        print(strides)

        y = [j for i, j in enumerate(orig_y) \
                if i - idx_first_y >= 0 and ((i - idx_first_y) % strides) == 0]

        y = np.array(y)
        data_length = len(y)

        x = []
        for i in range(data_length):
            start_idx = i * strides
            # end_idx does not contain -1 as ndarray indexing is ended before end_idx
            end_idx = start_idx + window_size
            x.append(orig_x[start_idx:end_idx:1])

        x = np.array(x)

        return x, y

        # for i, j in enumerate(orig_x):
        #     print(i, j)
        #
        # for i, j in enumerate(x):
        #     print(i, j)

if __name__ == '__main__':
    dt_cls = data_transform()
    df1 = pd.DataFrame(np.random.randn(20, 4), columns=['a', 'b', 'c', 'd'])

    dt_cls.create_window_data(0, df1, [1,3], 2, 3 ,2)

