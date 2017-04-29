#! /usr/bin/python3

import tensorflow as tf
import pandas as pd

import constant as ct
import config
import cnn as model1 
import k_means as model2 
import make_input

if __name__ == "__main__":

    # Load input data
    x1_height = 2
    num_y1_tpye = 2
    x1, x1_width, y1 = make_input.split_xy(
        csv_file_path="./input.csv",
        num_y_type=num_y1_tpye,
        x_height=x1_height)
    
    x2 = pd.read_csv("./input.csv")
    x2_width = len(x2.columns)

    # make each graph
    graph_cnn = tf.Graph()
    graph_k_means = tf.Graph()

    # setting session 
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    """
    # Model 1
    with tf.Session(graph=graph_cnn, config=session_conf) as sess:
        cnn = model1.CNN(model_name=config.MODEL1_NAME, session=sess)               
        
        cnn.create_model(
            x_height=x1_height,
            x_width=x1_width,
            num_NN_nodes=[2,3], 
            num_y_type=2, 
            filter_sizes=[[2,2],[1,2]], 
            num_filters=1)                  
        
        cnn.restore_all(model_name=config.MODEL1_NAME)
        cnn.train(
            x=x1,
            y=y1,
            dev_sample_percentage=0.1,
            batch_size=2,
            num_epochs=1,
            evaluate_every=2,
            saver_every=100)
        cnn.run(x1, y1) 
    """
    # Model 2
    with tf.Session(graph=graph_k_means, config=session_conf) as sess:
        k_means = model2.K_Means(model_name=config.MODEL2_NAME, session=sess)
        k_means.create_model(x2_width, config.NUM_CENTROID)
#        k_means.restore_all(config.MODEL2_NAME)
        k_means.train(x2, config.MAX_ITERS)
