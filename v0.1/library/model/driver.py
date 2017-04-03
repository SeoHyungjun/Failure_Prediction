#! /usr/bin/python3

import cnn
import tensorflow as tf
import set_out_dir

if __name__ == "__main__":
    cnn = cnn.CNN(directory="./")
    set_out_dir.make_dir("K-means,CNN,NN")
"""
    with tf.Graph().as_default():     
        cnn.create_model(
            input_size=[2,2], 
            num_NN_nodes=[2,3], 
            num_output=3, 
            filter_sizes=[[2,2],[1,2]], 
            num_filters=1)

        cnn.train(
            dev_sample_percentage=0.1,
            data_file_path="./",
            tag="tag",
            batch_size="20",
            num_epochs="10",
            evaluate_every="100",
            checkpoint_every="100")
    


    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        run_result = sess.run(cnn.test, feed_dict={cnn.input_x:[[[3,2],[5,4]], [[5,6],[8,9]]], cnn.input_y:[1,2]})

    print (run_result)
"""
