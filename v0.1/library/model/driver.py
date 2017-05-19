#! /usr/bin/python3

import tensorflow as tf
import pandas as pd

import constant as ct
#import cnn as model1 
import k_means as model2
import ann as model3
import make_input

if __name__ == "__main__":
    """
    # Load input data
    x1_height = 2
    num_y1_tpye = 2
    x1, x1_width, y1 = make_input.split_xy(
        csv_file_path="./input.csv",
        num_y_type=num_y1_tpye,
        x_height=x1_height)
    """
    x2 = pd.read_csv("/root/SMART/in_cluster/nor.csv")
#    x2 = pd.read_csv("./sample.csv")
    x2_width = len(x2.columns)

    num_y3_tpye = 2
    x3, x3_width, y3 = make_input.split_xy(
        csv_file_path="/root/SMART/in_ann/in_ann.csv",
        num_y_type=num_y3_tpye)

    # make each graph
    graph_cnn = tf.Graph()
    graph_k_means = tf.Graph()
    graph_ann = tf.Graph()

    # setting session 
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    """ 
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
    arg_dict2 = {'dir_model':'hihi'}
    # Model 2
    k_means = model2.K_Means()
    k_means.set_config(k_means, arg_dict2)
    k_means.set_x(k_means, x2)
    k_means.set_model_sequence(k_means, 2)
    k_means.create_model()
#    k_means.restore_all()
    k_means.train()
    k_means.run()
    

    arg_dict3 = {'tmp':''}
    # Model 3
    ann = model3.ANN()
    ann.set_config(ann, arg_dict3)
    ann.set_x(ann, x3)
    ann.set_y(ann, y3)
    ann.set_model_sequence(ann, 3)
    ann.create_model()
#    ann.restore_all()
    ann.train()

    result = ann.run()
#    with open("./out","w") as f:
#        [f.writelines(str(y)) for y in result[0]]
    print(result)
"""
        num_entire = len(result[0]) + 1
        num_correct = 0
        for i, prediction in enumerate(result[0]):
            if y3[i][0] == 1:
                real_value = 0
            elif y3[i][1] == 1:
                real_value = 1
            if real_value == prediction:
                num_correct = num_correct + 1
        accuracy = num_correct / num_entire
        print(accuracy)
    """
