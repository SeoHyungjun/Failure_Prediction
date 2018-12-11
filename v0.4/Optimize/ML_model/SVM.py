'''
import sklearn.svm

def make_SVM(C = 1, logGamma = -3 ):
    
    return sklearn.svm.SVC(C=C, gamma=10**logGamma)
'''
import os
import sys
from ML_model.base_ml import Machine_Learning

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sklearn.svm

class SVM(Machine_Learning):
    def __init__(self):
        pass

    def create_ml(self, C = 1, logGamma = -3, kernel = 'rbf'):
    #def create_ml(self, **kwarg):
        #C = kwarg['c']
        #logGamma = kwarg['loggamma']
        self.model = sklearn.svm.SVC(C=C, gamma = 10**logGamma, kernel = kernel)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def run(self, X):
        self.predictions = self.model.predict(X)

    def restore(self):
        pass
