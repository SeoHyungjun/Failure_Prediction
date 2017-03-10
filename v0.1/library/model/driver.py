#! /usr/bin/python3

import cnn
import tensorflow as tf

if __name__ == "__main__":
    cnn = cnn.CNN()
    cnn.create_model(
        input_size=[3,3], 
        num_NN_nodes=[2,3], 
        num_output=3, 
        filter_sizes=[[2,2],[1,2]], 
        num_filters=2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        run_result = sess.run(pooled_outputs, feed_dict={input_x:[[1,2,3],[4,5,6],[7,8,9]]})

    print (run_result)
