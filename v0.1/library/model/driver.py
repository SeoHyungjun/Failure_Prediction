#! /usr/bin/python3

import cnn as algorithm1 
import tensorflow as tf
import make_input

if __name__ == "__main__":

    # Load input data
    x_height = 2
    num_y_tpye = 2
    x, x_width, y = make_input.split_xy(
        csv_file_path="./input.csv",
        num_y_type=num_y_tpye,
        x_height=x_height)

    # make each graph
    graph_cnn = tf.Graph()

    # setting session 
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)


    # Model 1
    with tf.Session(graph=graph_cnn, config=session_conf) as sess:
        cnn = algorithm1.CNN(algorithm_name="CNN", session=sess)
                 
        cnn.create_model(
            x_height=x_height,
            x_width=x_width,
            num_NN_nodes=[2,3], 
            num_y_type=2, 
            filter_sizes=[[2,2],[1,2]], 
            num_filters=1)
                  
#        cnn.restore_all(algorithm_name="CNN")

        cnn.train(
            x=x,
            y=y,
            dev_sample_percentage=0.1,
            batch_size=2,
            num_epochs=3,
            evaluate_every=2,
            saver_every=100)
            
        cnn.run(x, y) 
          

    # Model 2
    with tf.Session(graph=graph_cnn, config=session_conf) as sess:
        pass
