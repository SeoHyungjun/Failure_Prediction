#! /usr/bin/python3

import cnn as model1 
import tensorflow as tf
import make_input

if __name__ == "__main__":

    # Load input data
    x, x_len, y = make_input.split_xy(csv_file_path="./input.csv", x_height=2, y_size=2)


    cnn = model1.CNN(directory="./")

    graph = tf.Graph()
    with graph.as_default():
        
        # setting session 
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

        with tf.Session(config=session_conf) as sess:
             
            cnn.restore(sess, graph, model_name="CNN")
            """
            cnn.create_model(
                x_height = 2,
                x_len = x_len,
                num_NN_nodes=[2,3], 
                y_size=2, 
                filter_sizes=[[2,2],[1,2]], 
                num_filters=1)
            """      
            cnn.train(
                x = x,
                y = y,
                sess = sess,
                dev_sample_percentage=0.1,
                model_name="CNN",
                batch_size=5,
                num_epochs=3,
                evaluate_every=100,
                saver_every=100)
            
               
