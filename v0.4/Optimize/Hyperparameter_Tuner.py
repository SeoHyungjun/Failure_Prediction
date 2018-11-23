import sys
from Optimization_function import PSO

import optunity  
from ML_model import SVM
import Nested_CV

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits

import numpy as np

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

class Hyperparameter_Tuner:
    def __init__(self, Opt_algorithm, Opt_hyperparameter, ML_algorithm):
        self.Opt_algorithm = Opt_algorithm
        self.Opt_hyperparameter = Opt_hyperparameter
        self.ML_algorithm = ML_algorithm

        self.iris_dataset = load_iris()
        self.iris_x, self.iris_X_test, self.iris_y, self.iris_y_test = train_test_split(self.iris_dataset['data'], self.iris_dataset['target'], random_state=0)

    def run_opt_algorithm(self):
        if self.Opt_algorithm == 'PSO':
            for ML_al in self.ML_algorithm.keys():
                #self.opt_function = 
                print('\nPSO ' + ML_al + ' train')
                #print(self.ML_algorithm[ML_al])
                self.solver = PSO.make_PSO(self.Opt_hyperparameter, self.ML_algorithm[ML_al])
                Opt_PSO_best = Nested_CV.nested_cv(self.solver, ML_al, self.ML_algorithm[ML_al], data, labels)
                print('Opt_PSO_best = ' + str(Opt_PSO_best))
                #self.nested_crossvalidation(self, ML_algorithm = ML_al, ML_hyperparameter = self.ML_algorithm[ML_al])

        elif self.Opt_algorithm == 'BO':
            for ML_al in self.ML_algorithm.keys():
                print('\nBO ' + ML_al + ' train')
                print(self.ML_algorithm[ML_al])
