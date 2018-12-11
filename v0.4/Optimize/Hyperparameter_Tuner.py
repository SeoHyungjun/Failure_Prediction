import sys
from Optimization_function import PSO

import optunity  
import Nested_CV

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits
import pandas as pd
from sklearn.utils import shuffle
#import random

import numpy as np
'''
digits = load_digits()
n = digits.data.shape[0]
positive_digit = 6
negative_digit = 8
positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
negative_idx = [i for i in range(n) if digits.target[i] == negative_digit]
# add some noise to the data to make it a little challenging
original_data = digits.data[positive_idx + negative_idx, ...]
data = original_data + 5 * np.random.randn(original_data.shape[0], original_data.shape[1])
labels = [True] * len(positive_idx) + [False] * len(negative_idx)

print(data)
print(labels)
'''
class Hyperparameter_Tuner:
    def __init__(self, Opt_algorithm, Opt_hyperparameter, ML_algorithm):
        self.Opt_algorithm = Opt_algorithm
        self.Opt_hyperparameter = Opt_hyperparameter
        self.ML_algorithm = ML_algorithm

        self.iris_dataset = load_iris()
        self.iris_x, self.iris_X_test, self.iris_y, self.iris_y_test = train_test_split(self.iris_dataset['data'], self.iris_dataset['target'], random_state=0)

    def load_data(self):
        failure_data = pd.read_csv("../data/h_failure/failure_96.csv", header = None, index_col=False)
        #failure_data = np.genfromtxt("../data/h_failure/failure_12.csv", delimiter=',',dtype=None)
        #print(failure_data)
        normal_data = pd.read_csv("../data/random_4/normal_4_data.csv", header = None, index_col=False)
        #normal_data = np.genfromtxt("../data/random_4/normal_4_data.csv", delimiter=',',dtype=None)
        #print(normal_data)
        normal_data = (shuffle(normal_data)).iloc[:len(failure_data)]

        #normal_data = normal_data.iloc[:len(failure_data)]    
        sum_data = pd.concat([failure_data, normal_data])
        sum_data = shuffle(sum_data)
        #print(sum_data.values)
        #sum_data.to_csv('sum_data.csv', index=False, header=False, columns=None )
        #sum_data =pd.read_csv('sum_data.csv', header = None, index_col = False)
        #print(sum_data)
        data = sum_data.iloc[:, 2:14]
        #print(data)
        labels = sum_data.iloc[:, 1:2]
        #print(labels)
        self.data = data.values.tolist()
        self.labels = labels[1].tolist()
        print(len(self.data))
        print(len(self.labels))
        print('load end')
        

    def run_opt_algorithm(self):
        if self.Opt_algorithm == 'PSO':
            for ML_al in self.ML_algorithm.keys():
                kernel_list = []
                #self.opt_function = 
                print('\nPSO ' + ML_al + ' train')
                #print(self.ML_algorithm[ML_al])
                if ML_al == 'SVM':
                    kernel_list = self.ML_algorithm[ML_al]['kernel']
                    self.ML_algorithm[ML_al].pop('kernel')
                self.solver = PSO.make_PSO(self.Opt_hyperparameter, self.ML_algorithm[ML_al])
                print(self.ML_algorithm[ML_al])
                Opt_PSO_best = Nested_CV.nested_cv(self.solver, ML_al, self.ML_algorithm[ML_al], self.data, self.labels, kernel_list)
                print('Opt_PSO_best_avg = ' + str(Opt_PSO_best))
                #self.nested_crossvalidation(self, ML_algorithm = ML_al, ML_hyperparameter = self.ML_algorithm[ML_al])

        elif self.Opt_algorithm == 'BO':
            for ML_al in self.ML_algorithm.keys():
                print('\nBO ' + ML_al + ' train')
                print(self.ML_algorithm[ML_al])
