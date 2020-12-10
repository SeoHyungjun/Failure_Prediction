import os
import sys
from ML_model.base_ml import Machine_Learning

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sklearn.ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

class GNB(Machine_Learning):
    def __init__(self):
        pass

    def create_ml(self, log_var_smoothing = -9):
        #print('n_estimators = ' + str(n_estimators) + ' max_depth = ' + str(max_depth) + ' max_features = ' + str(max_features)
        self.model = GaussianNB(var_smoothing = 10**log_var_smoothing)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def run(self, X):
        self.predictions = self.model.predict(X)

    def restore(self):
        pass
    
    def roc(self, y_test, x_test):
        '''
        self.fpr, self.tpr, thresholds = roc_curve(y_test, self.model.predict_proba(x_test)[:, 1])
        plt.plot(self.fpr, self.tpr, '-', label="gnb")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #plt.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('Fall-Out')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic example')
        plt.show()
        '''
        self.fpr, self.tpr, thresholds = precision_recall_curve(y_test, self.model.predict_proba(x_test)[:, 1])
        plt.plot(self.tpr, self.fpr, '-', label="svm")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #plt.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('Fall-Out')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic example')
        plt.show()
        
    def roc_store(self):
        pd.DataFrame(self.fpr).to_csv("roc/prc/1day/gnb_precision.csv", mode = 'a', header=None, index=None)
        pd.DataFrame(self.tpr).to_csv("roc/prc/1day/gnb_sigmoid_recall.csv", mode = 'a', header=None, index=None)
        '''
        pd.DataFrame(self.fpr).to_csv("roc/1day/gnb_roc_fpr.csv", mode = 'a', header=None, index=None)
        pd.DataFrame(self.tpr).to_csv("roc/1day/gnb_roc_tpr.csv", mode = 'a', header=None, index=None)
        '''