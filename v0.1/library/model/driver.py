#! /usr/bin/python3

import cnn as model1 
import tensorflow as tf

if __name__ == "__main__":

    cnn = model1.CNN(directory="./")

    graph = tf.Graph()
    with graph.as_default():
        
        cnn.restore(graph, model_name="CNN")
        """
        cnn.create_model(
            data_file_path="./input.csv",
            input_height = 2,
            num_NN_nodes=[2,3], 
            num_output=2, 
            filter_sizes=[[2,2],[1,2]], 
            num_filters=1)
        """
        
        cnn.train(
            dev_sample_percentage=0.1,
            model_name="CNN",
            batch_size=5,
            num_epochs=3,
            evaluate_every=100,
            saver_every=100)
    

"""
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        run_result = sess.run(cnn.test, feed_dict={cnn.input_x:[[[3,2],[5,4]], [[5,6],[8,9]]], cnn.input_y:[1,2]})

    print (run_result)
"""
