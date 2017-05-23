#! /usr/bin/python3

import sys
sys.path.insert(0, '../data_transform')

import tensorflow as tf
import pandas as pd
import numpy as np

import constant as ct
#import cnn as model1 
import k_means as model2
import ann as model3
import data_transform

if __name__ == "__main__":

    # 1. instantiate model calss
    k_means = model2.K_Means()
    ann = model3.ANN()

    # 2. set config
    arg_dict2 = {'dir_model':'hihi'}
    k_means.set_config(k_means, arg_dict2)
    arg_dict3 = {'tmp':''}
    ann.set_config(ann, arg_dict3)

    # 3. operate each model
    #   3-1 : read data
    #   3-2 : set_x,y,sequence 
    #   3-3 : execute model 
    ## k-means
    dt = data_transform.Data_transform()
    x2 = dt.read_csv("/root/SMART/in_cluster/nor.csv")
    k_means.set_x(k_means, x2)
    k_means.set_model_sequence(k_means, 2)
    k_means.create_model()
#    k_means.restore_all()
    k_means.train()
    k_means.run()
    ## ann
    data3 = dt.read_csv("/root/SMART/in_ann/in_ann.csv")
    x3, y3 = dt.split_xy_by_yindex(data3)
    ann.set_x(ann, x3)
    ann.set_y(ann, y3)
    ann.set_model_sequence(ann, 3)
    ann.create_model()
#    ann.restore_all()
    ann.train()
    result = ann.run()


#    print(result)
#    with open("./out","w") as f:
#        [f.writelines(str(y)) for y in result[0]]
    """
    # Load input data
    x1_height = 2
    num_y1_tpye = 2
    x1, x1_width, y1 = make_input.split_xy(
        csv_file_path="./input.csv",
        num_y_type=num_y1_tpye,
        x_height=x1_height)
    
     
    # Model 1
    with tf.Session(graph=graph_cnn, config=session_conf) as sess:
        cnn = model1.CNN(session=sess)               
        
        cnn.create_model(
            x_height=x1_height,
            x_width=x1_width,
            num_NN_nodes=[2,3], 
            num_y_type=2, 
            filter_sizes=[[2,2],[1,2]], 
            num_filters=1)                  
        
#        cnn.restore_all()
        cnn.train(
            x=x1,
            y=y1,
            dev_sample_percentage=0.1,
            batch_size=2,
            num_epochs=1,
            evaluate_interval=2,
            save_interval=100)
        cnn.run(x1, y1) 
    """

